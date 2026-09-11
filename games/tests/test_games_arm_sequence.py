"""Tests for the generic one-arm plan.

Three things here are worth more than the rest. The stage argv lists ARE the experiment, so they
are asserted against the registry rather than against strings repeated from the module. The
wall-clock caps are asserted to EXCEED the measured run time, because the bug this module was
written after was a cap sitting under it -- a cap that kills a healthy arm is worse than none, and
"180m" looked reasonable right up to the moment it truncated a five-hour sweep. And every stage's
flags are checked against the target CLI's own `--help`, which is the seam that costs the most to
discover late: on a rented box it is discovered after the meter has started.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from games import arm_sequence as plan
from games import plans, reward_spread
from games.arms import arm_game_ids
from games.generation import VLLM_COLOCATE_ENV
from games.s3_sync import DEFAULT_SYNC_COMMAND, build_sync_command
from games.screen_thinking import SCREEN_ROOT
from games.select_prompts import training_sampler
from games.tests.conftest import (
    OPTIONAL_KNOB_SETTINGS,
    OPTIONAL_SWITCH_SETTINGS,
    clear_every_optional_knob,
    clear_plan_independent_environment,
    offered_flags,
    set_every_optional_knob,
    used_flags,
)
from games.train import (
    ARMS,
    VLLM_GPU_FRACTION_ENV,
    VLLM_IS_CORRECTION_ENV,
    GameTrainConfig,
)

if TYPE_CHECKING:
    from games.stage_runner import Stage

# vLLM colocate at the production shape: ~13.5 minutes per step measured on a rented 95 GiB card
# (2026-08-19), so 70 steps land near 16 h.
MEASURED_COLOCATE_ARM_HOURS = 70 * 13.5 / 60.0
# The HF sweep's arithmetic for 64 prompts x 8 samples at the measured completion budget.
ESTIMATED_WORST_CASE_SWEEP_HOURS = 5.4

# What a launch gate greps a printed plan for, spaced as `plans.describe_stages` joins an argv.
PINNED_ESTIMATOR_STRINGS = (
    "--loss-type dapo",
    "--scale-rewards batch",
    "--acknowledge-liger-estimator-mismatch",
    "--micro-batch-size 1",
)


def cap_hours(cap: str) -> float:
    """Read a `timeout(1)` duration like "12h" or "180m" as hours."""
    scale = {"h": 1.0, "m": 1.0 / 60.0, "s": 1.0 / 3600.0}[cap[-1]]
    return float(cap[:-1]) * scale


def every_stage_built() -> list[Stage]:
    """Build every stage this plan can produce, one by one.

    A sweep and a stage that reads its corpus cannot share an invocation -- the corpus does not
    exist until the sweep has run -- so `stages()` can never return all four. The argv, log and
    artifact contracts are per stage rather than per invocation, so the tests that check all of them
    ask each builder directly.
    """
    return [builder() for builder in plan.STAGE_BUILDERS.values()]


@pytest.fixture(autouse=True)
def clean_plan_environment(
    monkeypatch: pytest.MonkeyPatch,
    exported_plan_independent_environment: None,  # exported before this fixture strips it
) -> None:
    """Strip every plan variable, so an operator's shell cannot change what these tests measure."""
    for name in (
        "ARM",
        "MODEL",
        "STAGES",
        "BACKEND",
        "CORPUS",
        "SWEEP_DIR",
        "SWEEP_GRADING",
        "SAMPLES_PER_PROMPT",
        "COMPLETION_TOKENS",
        "SCREEN_BUDGET",
        "SCREEN_TIMEOUT",
        "SWEEP_TIMEOUT",
        "ARM_TIMEOUT",
        "MAX_STEPS",
        "SAVE_STEPS",
        "GROUP",
        "PROMPTS_PER_STEP",
        "OUTPUT_DIR",
        "S3_DEST",
        "FROZEN_OPPONENT_MODEL",
        "FROZEN_OPPONENT_SAMPLES",
    ):
        monkeypatch.delenv(f"GAMES_ARM_SEQ_{name}", raising=False)
    clear_every_optional_knob(monkeypatch, plan.ENV_PREFIX)
    # Plan-independent by design (the box, and which arms a run is comparable to), so they need
    # stripping by their own names or an exporting shell changes every knob asserted here. The
    # old switch is stripped too: any value but "1" is refused at render time now, and a stray
    # "0" in the test shell would fail every stage build in this file. The fixture requested above
    # exports all four first, so a name dropped from the shared list turns this file red.
    clear_plan_independent_environment(monkeypatch)


@pytest.fixture
def hi_lo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the plan at the positive-control arm with one swept corpus on disk."""
    monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
    monkeypatch.setenv("GAMES_ARM_SEQ_OUTPUT_DIR", str(tmp_path / "run"))
    directory = tmp_path / "sweep"
    directory.mkdir()
    monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_DIR", str(directory))
    corpus = directory / "corpus-hi-lo-20260819T000000Z.jsonl"
    corpus.write_text('{"prompt": "x", "game_id": "hi-lo", "grading": "group-mix"}\n')
    return corpus


def write_frozen_corpus(directory: Path, *, opp_coop_prob: float | None) -> Path:
    """Write a vs-frozen corpus whose opponent column is filled, unfilled, or absent."""
    row: dict[str, object] = {"prompt": "x", "game_id": "pd-vs-frozen", "grading": "vs-fixed-mix"}
    if opp_coop_prob is not None:
        row["opp_coop_prob"] = opp_coop_prob
    corpus = directory / "corpus-pd-vs-frozen-20260819T000000Z.jsonl"
    corpus.write_text(json.dumps(row) + "\n")
    return corpus


class TestTheArmComesFromTheRegistryAndNowhereElse:
    """The registry defines what an arm is; a plan that also defines it is a plan that drifts."""

    def test_an_unset_arm_refuses_to_guess(self):
        with pytest.raises(ValueError, match="GAMES_ARM_SEQ_ARM is unset"):
            plan.arm_name()

    def test_an_unregistered_arm_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-self")
        with pytest.raises(ValueError, match="unknown arm"):
            plan.arm_name()

    def test_the_sweep_takes_its_game_and_grading_from_the_registry(self, hi_lo: Path):
        del hi_lo
        argv = plan.sweep_stage().argv
        assert argv[argv.index("--game") + 1] == ARMS["hi-lo-group"].game_id
        assert argv[argv.index("--grading") + 1] == ARMS["hi-lo-group"].grading

    def test_the_arm_trains_the_arm_that_was_named(self, hi_lo: Path):
        del hi_lo
        argv = plan.arm_stage().argv
        assert argv[argv.index("--arm") + 1] == "hi-lo-group"

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_every_registered_arm_can_build_a_sweep_without_a_gpu(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", name)
        monkeypatch.setenv("GAMES_ARM_SEQ_FROZEN_OPPONENT_MODEL", "some.model")
        assert plan.sweep_stage().argv


class TestWallClockCapsExceedTheMeasuredRunTime:
    """The bug this module answers: a cap under the healthy run time kills a working stage."""

    def test_the_hf_sweep_cap_clears_the_hf_sweep_estimate(self):
        assert cap_hours(plans.SWEEP_TIMEOUT_BY_BACKEND["hf"]) > ESTIMATED_WORST_CASE_SWEEP_HOURS

    def test_the_vllm_sweep_cap_is_shorter_but_still_generous(self):
        vllm, hf = plans.SWEEP_TIMEOUT_BY_BACKEND["vllm"], plans.SWEEP_TIMEOUT_BY_BACKEND["hf"]
        assert cap_hours(vllm) < cap_hours(hf)
        assert cap_hours(vllm) > ESTIMATED_WORST_CASE_SWEEP_HOURS / 3

    def test_the_colocate_arm_cap_clears_its_own_measurement(self):
        """Asserted against the measured per-step rate rather than against a number copied out of
        the module, so raising the cap without a measurement to justify it does not silently pass.
        """
        assert cap_hours(plans.COLOCATE_ARM_TIMEOUT) > MEASURED_COLOCATE_ARM_HOURS

    def test_the_arm_cap_is_the_colocate_cap_and_the_slow_export_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Rollouts are vLLM-only (owner decision 2026-08-26), so there is one cap to resolve --
        and the export that used to buy the 60h transformers cap fails the resolve outright.
        """
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv(VLLM_COLOCATE_ENV, "1")
        assert plan.arm_timeout() == plans.COLOCATE_ARM_TIMEOUT
        monkeypatch.setenv(VLLM_COLOCATE_ENV, "0")
        with pytest.raises(RuntimeError, match="2026-08-26"):
            plan.arm_timeout()

    def test_the_sweep_cap_follows_the_backend(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_BACKEND", "vllm")
        assert plan.sweep_timeout() == plans.SWEEP_TIMEOUT_BY_BACKEND["vllm"]
        monkeypatch.setenv("GAMES_ARM_SEQ_BACKEND", "hf")
        assert plan.sweep_timeout() == plans.SWEEP_TIMEOUT_BY_BACKEND["hf"]

    def test_the_arm_cap_does_not_follow_the_sweep_backend(self, monkeypatch: pytest.MonkeyPatch):
        """The two axes stay separate: `--backend` picks the sweep's engine, never the arm's cap.

        With one training backend left the conflation would be harmless today; the axes stay
        asserted apart so a future sweep backend cannot start leaking into the arm cap again.
        """
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_BACKEND", "hf")
        assert plan.arm_timeout() == plans.COLOCATE_ARM_TIMEOUT

    def test_an_unmeasured_backend_falls_back_to_the_slowest_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A too-long cap costs nothing; a too-short one kills a healthy sweep."""
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setattr(plan, "LOCAL_KINDS", frozenset({"hf", "vllm", "some-new-server"}))
        monkeypatch.setenv("GAMES_ARM_SEQ_BACKEND", "some-new-server")
        assert plan.sweep_timeout() == plans.SWEEP_TIMEOUT_BY_BACKEND["hf"]

    def test_an_operator_can_override_either_cap(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_TIMEOUT", "30m")
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM_TIMEOUT", "90m")
        assert plan.sweep_timeout() == "30m"
        assert plan.arm_timeout() == "90m"

    def test_every_gpu_stage_carries_a_cap(self, hi_lo: Path):
        del hi_lo
        for stage in (plan.screen_stage(), plan.sweep_stage(), plan.arm_stage()):
            assert stage.argv[0] == "timeout", stage.name
            assert cap_hours(stage.argv[1]) > 0, stage.name


class TestBackendSelection:
    """A sweep measures the checkpoint about to be trained, or it is not a baseline."""

    def test_the_default_is_the_hf_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        assert plan.backend() == "hf"

    @pytest.mark.parametrize("kind", ["bedrock", "mock", "codex"])
    def test_a_backend_that_cannot_sweep_the_policy_is_refused(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_BACKEND", kind)
        with pytest.raises(ValueError, match="cannot sweep a training baseline"):
            plan.backend()

    def test_the_chosen_backend_reaches_the_sweep_command(self, hi_lo: Path):
        del hi_lo
        argv = plan.sweep_stage().argv
        assert argv[argv.index("--backend") + 1] == "hf"


class TestASweepCannotShareAPlanWithAStageThatReadsItsCorpus:
    """`games.select_prompts` timestamps its output, so no plan can name the corpus in advance.

    Stage construction resolves the corpus eagerly and `stage_runner.load_plan` builds every stage
    before running any of them, so "sweep then arm" in one invocation has two outcomes and neither
    is the one it looks like: on a clean box it raises from the resolver before the sweep it was
    about to run, and with an older corpus present it pins the stale one, runs a fresh 5-12 hour
    sweep whose output nothing reads, and leaves the directory ambiguous for every later resolve.
    Both plans document two invocations; the refusal is what makes the module agree with them.
    """

    def test_the_default_plan_is_the_sweep_alone(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        assert plan.selected_stages() == ("sweep",)

    def test_sweeping_and_training_in_one_invocation_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_DIR", str(tmp_path / "empty"))
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "sweep,arm")
        with pytest.raises(ValueError, match="two invocations"):
            plan.stages()

    def test_sweeping_and_regrading_in_one_invocation_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "twin-pd-self")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "group-mix")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_DIR", str(tmp_path / "empty"))
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "sweep,regrade")
        with pytest.raises(ValueError, match="two invocations"):
            plan.stages()

    def test_the_refusal_still_fires_when_a_stale_corpus_would_have_resolved(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The expensive branch: resolution succeeds, so nothing else would have complained.

        The arm would have trained on the corpus already on disk while the sweep beside it wrote a
        second, newer one that nothing reads -- 5-12 GPU hours, no error, and a sweep directory
        left ambiguous for every later resolve. Asked for explicitly: no longer the default.
        """
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "sweep,arm")
        with pytest.raises(ValueError, match="two invocations"):
            plan.stages()

    def test_the_screen_may_still_ride_along_with_the_sweep(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """It reads no corpus, and running it in the same invocation saves a GPU handover."""
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_DIR", str(tmp_path / "empty"))
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "screen,sweep")
        assert [stage.name.split()[0] for stage in plan.stages()] == ["termination", "baseline"]

    def test_the_second_invocation_regrades_and_trains_together(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "regrade,arm")
        assert [stage.name.split()[0] for stage in plan.stages()] == ["regrade", "train"]


class TestStageSelection:
    def test_the_default_is_the_sweep(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        assert plan.selected_stages() == ("sweep",)

    def test_a_subset_keeps_plan_order_not_the_order_given(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm,screen")
        assert plan.selected_stages() == ("screen", "arm")

    def test_whitespace_is_tolerated(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", " sweep , arm ")
        assert plan.selected_stages() == ("sweep", "arm")

    def test_an_unknown_stage_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "sweep,evaluate")
        with pytest.raises(ValueError, match="unknown stage"):
            plan.selected_stages()

    def test_selecting_nothing_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", " , ")
        with pytest.raises(ValueError, match="selected nothing"):
            plan.selected_stages()

    def test_a_sweep_only_plan_needs_no_corpus(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_DIR", str(tmp_path / "empty"))
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "sweep")
        assert len(plan.stages()) == 1

    def test_only_the_regrade_stage_runs_off_the_gpu(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "regrade,arm")
        assert plan.screen_stage().needs_gpu
        assert plan.sweep_stage().needs_gpu
        assert plan.arm_stage().needs_gpu
        assert not plan.regrade_stage().needs_gpu


class TestOneSweepCanServeASecondGrading:
    """Sweeping twice would give two arms different prompts, which is a different experiment."""

    def test_by_default_the_sweep_is_graded_as_the_arm_is(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "twin-pd-self")
        assert plan.sweep_grading() == "self"
        assert not plan.regrade_required()

    def test_a_different_sweep_grading_implies_a_regrade(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "twin-pd-self")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "group-mix")
        assert plan.regrade_required()

    def test_an_unknown_sweep_grading_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "twin-pd-self")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "vibes")
        with pytest.raises(ValueError, match="is not a grading"):
            plan.sweep_grading()

    def test_the_arm_trains_on_the_regraded_copy(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        assert plan.arm_corpus() == hi_lo.with_name(f"{hi_lo.stem}-regraded-group-mix.jsonl")

    def test_the_regrade_writes_exactly_what_the_arm_reads(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "regrade,arm")
        regrade, train = plan.regrade_stage().argv, plan.arm_stage().argv
        assert regrade[regrade.index("--out") + 1] == train[train.index("--corpus") + 1]
        assert plan.regrade_stage().artifacts == (plan.arm_corpus(),)

    def test_a_plan_that_needs_a_regrade_and_omits_it_fails_at_build_time(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        with pytest.raises(ValueError, match="only the regrade stage writes"):
            plan.stages()

    def test_a_regrade_that_would_rewrite_the_corpus_unchanged_is_refused(self, hi_lo: Path):
        del hi_lo
        with pytest.raises(ValueError, match="already matches"):
            plan.regrade_stage()


class TestAVsFrozenArmCannotStartUngradeable:
    """`opp_coop_prob` outside [0,1] is what the corpus writes when no opponent was sampled."""

    def test_the_sweep_refuses_without_a_frozen_opponent_to_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "pd-vs-frozen")
        with pytest.raises(ValueError, match="FROZEN_OPPONENT_MODEL"):
            plan.sweep_stage()

    def test_the_frozen_opponent_reaches_the_sweep_command(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "pd-vs-frozen")
        monkeypatch.setenv("GAMES_ARM_SEQ_FROZEN_OPPONENT_MODEL", "some.frozen.model")
        argv = plan.sweep_stage().argv
        assert argv[argv.index("--frozen-opponent-model") + 1] == "some.frozen.model"
        assert argv[argv.index("--frozen-opponent-samples") + 1] == "8"

    def test_an_unfilled_opponent_column_stops_the_arm_before_the_gpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "pd-vs-frozen")
        monkeypatch.setenv("GAMES_ARM_SEQ_OUTPUT_DIR", str(tmp_path / "run"))
        monkeypatch.setenv(
            "GAMES_ARM_SEQ_CORPUS", str(write_frozen_corpus(tmp_path, opp_coop_prob=-1.0))
        )
        with pytest.raises(ValueError, match="outside"):
            plan.arm_stage()

    def test_a_missing_opponent_column_stops_the_arm_too(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "pd-vs-frozen")
        monkeypatch.setenv("GAMES_ARM_SEQ_OUTPUT_DIR", str(tmp_path / "run"))
        monkeypatch.setenv(
            "GAMES_ARM_SEQ_CORPUS", str(write_frozen_corpus(tmp_path, opp_coop_prob=None))
        )
        with pytest.raises(ValueError, match="no 'opp_coop_prob' column"):
            plan.arm_stage()

    def test_a_filled_opponent_column_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "pd-vs-frozen")
        monkeypatch.setenv("GAMES_ARM_SEQ_OUTPUT_DIR", str(tmp_path / "run"))
        monkeypatch.setenv(
            "GAMES_ARM_SEQ_CORPUS", str(write_frozen_corpus(tmp_path, opp_coop_prob=0.4))
        )
        assert plan.arm_stage().argv

    def test_the_check_leaves_other_gradings_alone(self, hi_lo: Path):
        del hi_lo
        plan.assert_opponent_probabilities_filled(plan.arm_corpus(), plan.game_arm())


class TestTheArmStageFollowsTheRepoRules:
    def test_resume_comes_with_an_explicit_output_dir(self, hi_lo: Path):
        """`latest` without one silently restarts at step 0 in a fresh timestamped directory."""
        del hi_lo
        argv = plan.arm_stage().argv
        assert argv[argv.index("--resume-from-checkpoint") + 1] == "latest"
        assert argv[argv.index("--output-dir") + 1] == str(plan.arm_output_dir())

    def test_it_promises_the_summary_that_proves_it_ran(self, hi_lo: Path):
        del hi_lo
        assert plan.arm_stage().artifacts == (plan.arm_output_dir() / "train_summary.json",)

    def test_thinking_stays_on_at_the_measured_budget(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_MODEL", "Qwen/Qwen3.5-2B")
        argv = plan.arm_stage().argv
        assert "--no-thinking" not in argv
        assert "--allow-short-completions" not in argv
        assert argv[argv.index("--max-completion-tokens") + 1] == "24576"

    def test_the_sweep_samples_at_the_training_sampler(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Asserted against `select_prompts.TRAINING_SAMPLER`, not against the literals.

        Pinning '1.0'/'1.0'/'0' pins the flags against themselves: if TRL's generation defaults
        move or `games.train`'s sampler fields are retuned, the sweep would keep selecting prompts
        off the policy that trains, silently, with a suite still green. The whole contrast rests on
        the sweep sampling the trainer's own policy, so the two have to be tied.
        """
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_MODEL", "Qwen/Qwen3.5-2B")
        sampler = training_sampler("Qwen/Qwen3.5-2B")
        argv = plan.sweep_stage().argv
        assert "--thinking" in argv
        assert argv[argv.index("--temperature") + 1] == str(sampler.temperature)
        assert argv[argv.index("--top-p") + 1] == str(sampler.top_p)
        assert argv[argv.index("--top-k") + 1] == str(sampler.top_k)
        assert argv[argv.index("--max-new-tokens") + 1] == str(sampler.max_new_tokens)

    def test_the_training_sampler_is_the_sampler_the_trainer_actually_uses(self):
        """Otherwise the tie above is to a constant that has drifted away from the trainer."""
        trainer_defaults = GameTrainConfig(arm="hi-lo-group", corpus_path="unused.jsonl")
        sampler = training_sampler(trainer_defaults.model_id)
        assert trainer_defaults.temperature == sampler.temperature
        assert trainer_defaults.top_p == sampler.top_p
        assert trainer_defaults.top_k == sampler.top_k
        assert trainer_defaults.max_completion_tokens == sampler.max_new_tokens

    def test_the_budget_follows_the_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_MODEL", "Qwen/Qwen3.5-9B")
        nine_b = int(plan.completion_tokens())
        monkeypatch.setenv("GAMES_ARM_SEQ_MODEL", "Qwen/Qwen3.5-2B")
        assert nine_b > int(plan.completion_tokens())

    def test_it_checkpoints_often_enough_for_a_ladder(self, hi_lo: Path):
        del hi_lo
        argv = plan.arm_stage().argv
        save_steps = int(argv[argv.index("--save-steps") + 1])
        assert int(argv[argv.index("--max-steps") + 1]) // save_steps >= 10

    def test_the_arm_ships_to_its_own_s3_prefix(self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_S3_DEST", "s3://bucket/games_rl/wave1/")
        stage_env = plan.arm_stage().env
        assert stage_env is not None
        assert stage_env["GAMES_S3_DEST"].endswith("hi-lo-group-qwen35-2b")

    def test_the_sync_stays_off_when_no_destination_is_set(self, hi_lo: Path):
        del hi_lo
        assert plan.arm_stage().env is None

    def test_a_missing_corpus_fails_before_any_gpu_is_reserved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_CORPUS", str(tmp_path / "absent.jsonl"))
        with pytest.raises(FileNotFoundError, match="does not exist"):
            plan.arm_stage()


class TestTheEngineKnobsAreVisibleInThePrintedCommand:
    """`--print-plan` is the pre-launch check, so it has to say what the engine will hold.

    The trainer reads the same variables on its own, so a plan that stayed silent would still run
    correctly -- but an operator checking the command before starting the meter could not see the
    engine's card share or an estimator caveat. There is no backend flag left to render: rollouts
    are vLLM-only, and a stage build on a box exporting otherwise refuses instead of rendering.
    """

    def test_the_measured_fraction_is_rendered_with_nothing_exported(self, hi_lo: Path):
        del hi_lo
        argv = plan.arm_stage().argv
        assert "--vllm-colocate" not in argv
        assert float(argv[argv.index("--vllm-gpu-memory-utilization") + 1]) == pytest.approx(0.35)

    def test_an_operator_s_fraction_reaches_the_rendered_command(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv(VLLM_GPU_FRACTION_ENV, "0.5")
        argv = plan.arm_stage().argv
        assert float(argv[argv.index("--vllm-gpu-memory-utilization") + 1]) == pytest.approx(0.5)

    def test_dropping_the_estimator_correction_is_rendered_rather_than_left_to_the_environment(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The caveat this carries is on the run's absolute levels, so it belongs in the command."""
        del hi_lo
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "0")
        assert "--no-vllm-importance-sampling-correction" in plan.arm_stage().argv

    def test_the_correction_flag_stays_off_the_command_while_it_is_on(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "1")
        assert "--no-vllm-importance-sampling-correction" not in plan.arm_stage().argv

    def test_a_stage_build_refuses_on_a_box_exporting_the_slow_path(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The exact environment that bought the 2026-08-25 slow arm cannot render a stage."""
        del hi_lo
        monkeypatch.setenv(VLLM_COLOCATE_ENV, "0")
        with pytest.raises(RuntimeError, match="2026-08-26"):
            plan.arm_stage()


class TestCorpusResolution:
    """The shared resolver refuses to guess; these are the ways it has been wrong before."""

    def test_it_returns_the_single_non_empty_corpus(self, hi_lo: Path):
        assert plan.resolve_corpus() == hi_lo

    def test_it_raises_when_the_sweep_never_ran(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_DIR", str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError, match="never ran"):
            plan.resolve_corpus()

    def test_it_raises_on_an_empty_corpus_rather_than_training_on_nothing(self, hi_lo: Path):
        hi_lo.write_text("")
        with pytest.raises(ValueError, match=r"every corpus .* is empty"):
            plan.resolve_corpus()

    def test_it_refuses_to_pick_between_two_corpora(self, hi_lo: Path):
        hi_lo.with_name("corpus-hi-lo-20260819T010000Z.jsonl").write_text("{}\n")
        with pytest.raises(ValueError, match="non-empty corpora"):
            plan.resolve_corpus()

    def test_a_regraded_sibling_is_not_a_second_candidate(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The regrade writes beside its source, and its name used to start with `corpus-` too.

        Once the regrade had run, the directory held two non-empty `corpus-*.jsonl` and every later
        resolve refused -- so a regraded arm could no longer resolve its own corpus, blocking both
        `--print-corpus` and any resume from a shell without GAMES_ARM_SEQ_CORPUS exported.
        """
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        regraded = plan.arm_corpus()
        regraded.write_text('{"prompt": "x", "grading": "group-mix"}\n')
        assert regraded.parent == hi_lo.parent
        assert plan.resolve_corpus() == hi_lo
        assert plan.swept_corpus() == hi_lo

    def test_an_explicit_corpus_overrides_resolution(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        del hi_lo
        chosen = tmp_path / "chosen.jsonl"
        monkeypatch.setenv("GAMES_ARM_SEQ_CORPUS", str(chosen))
        assert plan.swept_corpus() == chosen

    def test_the_sweep_directory_is_per_arm_and_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_MODEL", "Qwen/Qwen3.5-2B")
        first = plan.sweep_dir()
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "chicken-group")
        assert plan.sweep_dir() != first


class TestTheRewardSpreadHook:
    """Mixing payoff variants of different spread is a decision, so it has to be surfaced.

    Asked of `games.reward_spread`, which owns the table now: both this plan and the contrast pair log
    it before their sweep, and the pair used to reach it by importing this module -- which dragged the
    whole training stack into the headline experiment's `--print-corpus`. The plan still calls it, so
    the caplog logger has to be the one that actually emits, or the absence assertion below would pass
    for the wrong reason.
    """

    def test_a_multi_variant_arm_warns_that_its_rungs_train_unequally(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "stag-hunt-group")
        with caplog.at_level(logging.WARNING, logger=reward_spread.logger.name):
            plan.log_reward_spread(plan.game_arm())
        assert "mixes payoff variants" in caplog.text

    def test_a_single_variant_arm_does_not(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        with caplog.at_level(logging.WARNING, logger=reward_spread.logger.name):
            plan.log_reward_spread(plan.game_arm())
        assert "mixes payoff variants" not in caplog.text

    def test_the_stag_ladder_spans_the_ratio_the_warning_is_about(self):
        """At a mostly-cooperating mix the four rungs differ by roughly an order of magnitude."""
        spreads = reward_spread.variant_reward_spreads(ARMS["stag-hunt-group"], 0.9)
        assert len(spreads) == 4
        assert max(spreads.values()) / min(spreads.values()) > 5.0

    def test_the_compression_would_be_hidden_by_aggregating_over_mixes(self):
        """Why the spread is per mix: the rung narrowest at 0.9 is the widest at 0.1."""
        at_low = reward_spread.variant_reward_spreads(ARMS["stag-hunt-group"], 0.1)
        at_high = reward_spread.variant_reward_spreads(ARMS["stag-hunt-group"], 0.9)
        narrowest_at_high = min(at_high, key=lambda name: at_high[name])
        assert at_low[narrowest_at_high] == max(at_low.values())

    def test_a_grading_with_no_opponent_expectation_has_no_spread_to_report(self):
        assert reward_spread.variant_reward_spreads(ARMS["dictator"], 0.5) == {}


class TestScreenArtifactsLandWhereEverythingElseLooksForThem:
    """One screen directory, and it is the one `games.screen_thinking` writes to by default.

    The plans said `artifacts/games/screens` (plural) while the writer's own default, the provenance
    paths in `games/termination.py`, and every screen artifact on disk say `artifacts/games/screen`.
    A screen launched through a plan therefore landed in a sibling directory nothing else reads.
    """

    def test_the_screen_stage_writes_where_screen_thinking_writes(self, hi_lo: Path):
        del hi_lo
        (artifact,) = plan.screen_stage().artifacts
        assert artifact.parent == plans.REPO / SCREEN_ROOT

    def test_the_promised_artifact_is_the_path_the_command_is_told_to_write(self, hi_lo: Path):
        del hi_lo
        stage = plan.screen_stage()
        argv = stage.argv
        assert stage.artifacts == (Path(argv[argv.index("--json-out") + 1]),)


class TestEveryStageLogShipsWithTheRun:
    """Each stage's log must land inside the directory the run's S3 sync already covers.

    The sync ships the run directory wholesale and nothing in the repo ships
    `artifacts/games/logs/`, so a log written only there survives exactly as long as the box does.
    Operators were papering over that with a hand-rolled `aws s3 sync artifacts/games/logs/`, and on
    2026-08-19 a run's only surviving record was such a log-only sync. The assertion is on coverage
    rather than on a path string: the sync's source is the run directory, no filter flag stands
    between it and the log, so the log ships.
    """

    def test_every_stage_writes_a_log_the_run_directory_sync_would_ship(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "regrade,arm")
        run_dir = plan.arm_output_dir()
        argv = build_sync_command(run_dir, "s3://bucket/runs/hi-lo-group")
        assert argv[len(DEFAULT_SYNC_COMMAND)] == str(run_dir)
        assert not [flag for flag in argv if flag.startswith(("--exclude", "--include"))]
        for stage in every_stage_built():
            shipped = [path for path in stage.log_destinations() if path.is_relative_to(run_dir)]
            assert shipped, f"{stage.name} writes no log inside the directory that ships"

    def test_the_shared_log_directory_still_gets_its_copy(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "regrade,arm")
        # An operator tails these, and `resolve_single_corpus` names one in its failure message.
        for stage in every_stage_built():
            assert stage.log_path is not None
            assert stage.log_path.parent == plans.LOG_DIR
            assert stage.log_path in stage.log_destinations()


class TestThePlanEntryPoint:
    """`stages()` and the print modes are what an operator and stage_runner both depend on."""

    def test_the_default_plan_is_the_sweep_alone(self, hi_lo: Path):
        """The corpus is not knowable until the sweep has run, so the default cannot go further."""
        del hi_lo
        built = plan.stages()
        assert [stage.name.split()[0] for stage in built] == ["baseline"]

    def test_the_two_invocations_cover_every_stage_between_them(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "screen,sweep")
        first = [stage.name.split()[0] for stage in plan.stages()]
        plan.arm_corpus().write_text('{"prompt": "x", "grading": "group-mix"}\n')
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "regrade,arm")
        second = [stage.name.split()[0] for stage in plan.stages()]
        assert first + second == ["termination", "baseline", "regrade", "train"]
        assert hi_lo.is_file()

    def test_run_mode_maps_a_verified_sequence_to_exit_zero(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """`--plan` is the documented launch path, but this branch is an operator's fallback.

        The mapping itself is what matters: a shell, a tmux session or an operator at 2am has to
        tell success from failure without reading the log.
        """
        del hi_lo
        monkeypatch.setattr(plan, "stages", list)
        monkeypatch.setattr(plan, "run_sequence", lambda _stages: SimpleNamespace(ok=True))
        assert plan.main([]) == 0
        monkeypatch.setattr(plan, "run_sequence", lambda _stages: SimpleNamespace(ok=False))
        assert plan.main([]) == 1

    def test_print_corpus_resolves_without_running_anything(
        self, hi_lo: Path, capsys: pytest.CaptureFixture[str]
    ):
        assert plan.main(["--print-corpus"]) == 0
        assert capsys.readouterr().out.strip() == str(hi_lo)

    def test_print_plan_renders_the_sweep_it_is_about_to_run(
        self, hi_lo: Path, capsys: pytest.CaptureFixture[str]
    ):
        del hi_lo
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        assert "hi-lo-group" in printed
        assert "games.select_prompts" in printed

    def test_print_plan_renders_the_arm_and_its_promised_artifact(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """The second invocation is the expensive one, so it is the one worth reading first.

        Asked for by name rather than through the default plan, because the sweep and the arm cannot
        share an invocation: the sweep's corpus has no knowable path until it has run.
        """
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        assert "hi-lo-group" in printed
        assert "games.train" in printed
        assert "train_summary.json" in printed

    def test_print_plan_carries_the_estimator_pin_a_launch_gate_greps_for(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """The printout is where a launch script checks it got the estimator it asked for.

        This plan asks for nothing per arm -- it builds its stage through `plans.build_arm_stage`
        like the other two -- so the pin reaching the printout is the whole per-plan wiring.
        """
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        monkeypatch.setenv(plans.ESTIMATOR_PIN_ENV, plans.FLAGSHIP_ESTIMATOR_PIN)
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        for pinned in PINNED_ESTIMATOR_STRINGS:
            assert pinned in printed, pinned

    def test_print_plan_names_no_estimator_at_all_when_no_pin_is_asked_for(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """An unpinned launch trains under the repo defaults, which the printout must not contradict."""
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        for flag in ("--loss-type", "--scale-rewards", "--micro-batch-size"):
            assert flag not in printed, flag
        assert "--acknowledge-liger-estimator-mismatch" not in printed

    def test_print_plan_refuses_a_misspelled_pin_rather_than_printing_a_plan(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A silent typo would print a green-looking plan and train the defaults under a pinned name."""
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        monkeypatch.setenv(plans.ESTIMATOR_PIN_ENV, "flagship-dapo-batch")
        with pytest.raises(ValueError, match=plans.ESTIMATOR_PIN_ENV):
            plan.main(["--print-plan"])


class TestEveryStageFlagIsAcceptedByItsCli:
    """A plan builds argv as data, so a flag renamed upstream is invisible until the stage runs.

    On a rented instance it runs after the meter has started, an hour into a bootstrap. Asking each
    CLI what it accepts costs ten seconds per module here, once per session (`games.tests.conftest`).
    """

    def _assert_accepted(self, argv: tuple[str, ...]) -> None:
        module, used = used_flags(argv)
        unknown = used - offered_flags(module)
        assert not unknown, f"{module} does not accept {sorted(unknown)}"

    def test_every_stage_the_plan_can_build(self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "self")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "regrade,arm")
        for stage in every_stage_built():
            self._assert_accepted(stage.argv)

    def test_a_vs_frozen_sweep(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "pd-vs-frozen")
        monkeypatch.setenv("GAMES_ARM_SEQ_FROZEN_OPPONENT_MODEL", "some.frozen.model")
        self._assert_accepted(plan.sweep_stage().argv)

    def test_a_colocate_arm_with_the_estimator_correction_dropped(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Every colocate flag at once, since none of them is rendered on the default path."""
        del hi_lo
        monkeypatch.setenv(VLLM_COLOCATE_ENV, "1")
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "0")
        self._assert_accepted(plan.arm_stage().argv)

    def test_a_pinned_arm(self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch):
        """The pinned four are off the default path too, so they get their own pass against --help."""
        del hi_lo
        monkeypatch.setenv(plans.ESTIMATOR_PIN_ENV, plans.FLAGSHIP_ESTIMATOR_PIN)
        self._assert_accepted(plan.arm_stage().argv)

    def test_an_arm_with_every_optional_knob_set(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The five optional knobs and the two instrument switches are off the default path, so
        they need their own pass too."""
        del hi_lo
        set_every_optional_knob(monkeypatch, plan.ENV_PREFIX)
        argv = plan.arm_stage().argv
        for _suffix, flag in OPTIONAL_SWITCH_SETTINGS:
            assert flag in argv, flag
        self._assert_accepted(argv)


class TestTheOptionalTrainingKnobs:
    """Five training knobs a plan renders only when its own environment sets one.

    Absence is the load-bearing half. Every plan and every kit written before these knobs existed
    must render the argv it always did, so `games/train.py`'s argparse defaults stay in force and no
    banked arm's treatment moves because a knob arrived. Two of the five, `--adam-epsilon` and
    `--lora-dropout`, are treatment changes against the banked 9B pair, which is why they are
    exported knobs recorded in `run_config.json` rather than edited defaults.

    `--lr-scheduler` is checked against transformers' own scheduler names here, at plan time: on the
    rented box a misspelling would be discovered after the meter had started.
    """

    def test_none_of_them_is_rendered_when_the_environment_sets_none(self, hi_lo: Path):
        del hi_lo
        argv = plan.arm_stage().argv
        for _suffix, flag, _value in OPTIONAL_KNOB_SETTINGS:
            assert flag not in argv, flag

    def test_each_one_reaches_the_command_with_the_value_that_was_set(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        set_every_optional_knob(monkeypatch, plan.ENV_PREFIX)
        argv = plan.arm_stage().argv
        for _suffix, flag, value in OPTIONAL_KNOB_SETTINGS:
            assert flag in argv, flag
            assert argv[argv.index(flag) + 1] == value, flag

    def test_setting_one_knob_leaves_the_other_four_off_the_command(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv(f"{plan.ENV_PREFIX}ADAM_EPSILON", "1e-15")
        argv = plan.arm_stage().argv
        assert argv[argv.index("--adam-epsilon") + 1] == "1e-15"
        for flag in ("--parse-penalty", "--lr-scheduler", "--warmup-ratio", "--lora-dropout"):
            assert flag not in argv, flag

    def test_an_unknown_scheduler_name_is_refused_rather_than_rendered(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A typo would otherwise reach the box and die there, an hour of bootstrap later."""
        del hi_lo
        monkeypatch.setenv(f"{plan.ENV_PREFIX}LR_SCHEDULER", "constant_with_warumup")
        with pytest.raises(ValueError, match=f"{plan.ENV_PREFIX}LR_SCHEDULER"):
            plan.arm_stage()

    def test_the_refusal_reaches_the_printed_plan(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """`--print-plan` is the cheap check before the meter starts, so it has to be the one that
        fails rather than printing a green-looking plan."""
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        monkeypatch.setenv(f"{plan.ENV_PREFIX}LR_SCHEDULER", "cosine_with_warmpup")
        with pytest.raises(ValueError, match=f"{plan.ENV_PREFIX}LR_SCHEDULER"):
            plan.main(["--print-plan"])

    def test_every_scheduler_transformers_offers_is_accepted(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        for name in sorted(plans.LR_SCHEDULER_NAMES):
            monkeypatch.setenv(f"{plan.ENV_PREFIX}LR_SCHEDULER", name)
            argv = plan.arm_stage().argv
            assert argv[argv.index("--lr-scheduler") + 1] == name

    def test_a_non_numeric_value_is_refused_naming_its_own_variable(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The kit exports these as strings, so a stray character is a plan-time question."""
        del hi_lo
        monkeypatch.setenv(f"{plan.ENV_PREFIX}WARMUP_RATIO", "0.O5")
        with pytest.raises(ValueError, match=f"{plan.ENV_PREFIX}WARMUP_RATIO"):
            plan.arm_stage()

    def test_a_negative_parse_penalty_and_an_exponent_are_both_numeric(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The two shapes these knobs actually take: a signed penalty and an exponent epsilon."""
        del hi_lo
        monkeypatch.setenv(f"{plan.ENV_PREFIX}PARSE_PENALTY", "-0.25")
        monkeypatch.setenv(f"{plan.ENV_PREFIX}ADAM_EPSILON", "1e-15")
        argv = plan.arm_stage().argv
        assert argv[argv.index("--parse-penalty") + 1] == "-0.25"
        assert argv[argv.index("--adam-epsilon") + 1] == "1e-15"

    def test_the_printed_plan_names_each_knob_the_launch_gate_greps_for(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        set_every_optional_knob(monkeypatch, plan.ENV_PREFIX)
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        for _suffix, flag, value in OPTIONAL_KNOB_SETTINGS:
            assert f"{flag} {value}" in printed, flag

    def test_the_printed_plan_names_no_knob_when_none_is_set(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """A plan that set nothing must not read as though it set a default."""
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        assert "knob" not in printed
        for _suffix, flag, _value in OPTIONAL_KNOB_SETTINGS:
            assert flag not in printed, flag


class TestThePrintedPlanNamesTheGameSet:
    """The launch gate greps the printed plan, so a breadth arm's corpus games have to be in it.

    An arm's header line names one game, which was the whole story until an arm's corpus carried six.
    A kit launching the breadth arm has one cheap chance to notice it is about to train the wrong
    corpus, and that is `--print-plan` before the meter starts: the games line is what its gate pins.
    """

    def games_line(self, capsys: pytest.CaptureFixture[str]) -> str:
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        return next(line for line in printed.splitlines() if line.startswith("games "))

    def test_a_breadth_arm_prints_its_whole_game_set_in_order(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "prosocial-breadth-care1")
        monkeypatch.setenv("GAMES_ARM_SEQ_OUTPUT_DIR", str(tmp_path / "run"))
        assert self.games_line(capsys) == (
            "games       twin-pd, pd-reskin, stag-hunt, chicken, public-goods, trust-vs-stated-return"
        )

    def test_a_single_game_arm_prints_that_one_game(
        self, hi_lo: Path, capsys: pytest.CaptureFixture[str]
    ):
        del hi_lo
        assert self.games_line(capsys) == "games       hi-lo"

    def test_the_line_is_the_registrys_set_for_every_arm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        # Asserted against the registry rather than against strings repeated from the module, the way
        # every other stage assertion in this file is.
        monkeypatch.setenv("GAMES_ARM_SEQ_FROZEN_OPPONENT_MODEL", "some.model")
        for name, arm in sorted(ARMS.items()):
            monkeypatch.setenv("GAMES_ARM_SEQ_ARM", name)
            monkeypatch.setenv("GAMES_ARM_SEQ_OUTPUT_DIR", str(tmp_path / "run"))
            expected = ", ".join(arm_game_ids(arm))
            assert self.games_line(capsys).split(None, 1)[1] == expected, name


class TestTheOptionalTrainingSwitches:
    """The valueless flags rendered only where the environment sets the variable to exactly "1".

    The wave-4b probe runs the vLLM importance-sampling correction ON in log-only mode with the fp32
    head, so the sampler-versus-trainer mismatch is measured on five steps without changing their
    gradient; the training arm runs the correction OFF as the banked pair did. Both roles launch
    through this plan from one kit, so the switches have to be exports, rendered only when asked for,
    and refused where the trainer would refuse them.

    `ALLOW_SHORT_COMPLETIONS` is the third row and belongs to neither role: it renders the trainer's
    escape hatch from the measured completion floor, for the plumbing smoke and the timing probe. Its
    tests are below, because what has to hold for it is the opposite of the instrument pair's -- it
    renders beside the correction OFF rather than being refused there.
    """

    def test_none_is_rendered_when_the_environment_sets_none(self, hi_lo: Path):
        del hi_lo
        argv = plan.arm_stage().argv
        for _suffix, flag in OPTIONAL_SWITCH_SETTINGS:
            assert flag not in argv, flag

    def test_each_reaches_the_command_as_one_token_when_set_to_1(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        for suffix, _flag in OPTIONAL_SWITCH_SETTINGS:
            monkeypatch.setenv(f"{plan.ENV_PREFIX}{suffix}", "1")
        argv = plan.arm_stage().argv
        for _suffix, flag in OPTIONAL_SWITCH_SETTINGS:
            assert argv.count(flag) == 1, flag
            assert argv[argv.index(flag) + 1].startswith("--"), flag

    @pytest.mark.parametrize("chosen", [flag for _suffix, flag in OPTIONAL_SWITCH_SETTINGS])
    def test_setting_one_leaves_every_other_switch_off_the_command(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, chosen: str
    ):
        del hi_lo
        suffix = next(suffix for suffix, flag in OPTIONAL_SWITCH_SETTINGS if flag == chosen)
        monkeypatch.setenv(f"{plan.ENV_PREFIX}{suffix}", "1")
        argv = plan.arm_stage().argv
        assert chosen in argv
        for _suffix, flag in OPTIONAL_SWITCH_SETTINGS:
            if flag != chosen:
                assert flag not in argv, flag

    @pytest.mark.parametrize("suffix", [suffix for suffix, _flag in OPTIONAL_SWITCH_SETTINGS])
    def test_a_value_other_than_a_bare_1_is_refused_naming_its_variable(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
    ):
        del hi_lo
        monkeypatch.setenv(f"{plan.ENV_PREFIX}{suffix}", "true")
        with pytest.raises(ValueError, match=f"{plan.ENV_PREFIX}{suffix}"):
            plan.arm_stage()

    def test_an_instrument_switch_beside_the_correction_off_is_refused_before_the_meter_starts(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The training role exports the correction OFF; a kit that copied the probe's switches into
        it would otherwise die on the box, after the bootstrap, with the trainer's own refusal."""
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "0")
        monkeypatch.setenv(f"{plan.ENV_PREFIX}IS_LOG_ONLY", "1")
        with pytest.raises(ValueError, match=VLLM_IS_CORRECTION_ENV):
            plan.main(["--print-plan"])

    def test_print_plan_carries_a_knob_line_a_launch_gate_can_grep_for(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """The probe's plan gate asserts its flags PRESENT and the training arm's asserts them
        ABSENT, so the printout has to say which it rendered."""
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        for suffix, _flag in OPTIONAL_SWITCH_SETTINGS:
            monkeypatch.setenv(f"{plan.ENV_PREFIX}{suffix}", "1")
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        for _suffix, flag in OPTIONAL_SWITCH_SETTINGS:
            assert f"knob        {flag}\n" in printed, flag
            assert f" {flag} " in printed or printed.rstrip().endswith(flag), flag

    def test_print_plan_names_none_of_them_when_none_is_asked_for(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        for _suffix, flag in OPTIONAL_SWITCH_SETTINGS:
            assert flag not in printed, flag


class TestTheShortCompletionsSwitchIsForSmokesAndTimingProbes:
    """`GAMES_ARM_SEQ_ALLOW_SHORT_COMPLETIONS=1` renders `--allow-short-completions`.

    `games.train` refuses a thinking-ON run below the measured termination floor, and this is the flag
    that says "I know, this run is a probe". The launch it exists for here is the local plumbing smoke
    of the wave-4b arm: two steps of a 2B on the shared L4 inside the limiter's 15-minute cap, which
    the arm's own 32,768-token budget cannot finish once the corpus carries trust rows that deliberate
    to the cap, against a repo rule that caps local GPU work at about ten minutes.

    Two things it must not do. It must not be refused beside the correction OFF, which the training
    role and therefore its smoke both export -- the instrument pair's coupling is the instrument
    pair's, not the table's. And it must not go unrecorded: `games.train` carries the field into
    `run_config.json`, so a smoke's launch record says it was a smoke, and this plan's printed knob
    line is what a kit's plan gate greps to require the flag ABSENT from its TRAIN and PROBE roles.
    """

    def test_it_renders_beside_the_correction_the_training_role_turns_off(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del hi_lo
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "0")
        monkeypatch.setenv(f"{plan.ENV_PREFIX}ALLOW_SHORT_COMPLETIONS", "1")
        argv = plan.arm_stage().argv
        assert argv.count("--allow-short-completions") == 1
        assert "--no-vllm-importance-sampling-correction" in argv

    def test_the_printed_plan_names_it_beside_the_correction_off(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """The whole printout, not just the stage build: a kit's plan gate reads this text."""
        del hi_lo
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "arm")
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "0")
        monkeypatch.setenv(f"{plan.ENV_PREFIX}ALLOW_SHORT_COMPLETIONS", "1")
        assert plan.main(["--print-plan"]) == 0
        assert "knob        --allow-short-completions\n" in capsys.readouterr().out

    def test_the_budget_is_still_the_plans_own_because_the_switch_only_permits_a_short_one(
        self, hi_lo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The switch is permission, not a budget: a smoke lowers `COMPLETION_TOKENS` itself, and a
        switch that quietly moved the budget would make every probe's own cap unreadable."""
        del hi_lo
        monkeypatch.setenv(f"{plan.ENV_PREFIX}MODEL", "Qwen/Qwen3.5-2B")
        monkeypatch.setenv(f"{plan.ENV_PREFIX}ALLOW_SHORT_COMPLETIONS", "1")
        argv = plan.arm_stage().argv
        assert argv[argv.index("--max-completion-tokens") + 1] == "24576"
        monkeypatch.setenv(f"{plan.ENV_PREFIX}COMPLETION_TOKENS", "2048")
        smoke = plan.arm_stage().argv
        assert smoke[smoke.index("--max-completion-tokens") + 1] == "2048"
        assert "--allow-short-completions" in smoke
