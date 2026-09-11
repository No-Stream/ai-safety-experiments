"""Pin the compute-matched non-game control: what it is matched to, and that a mismatch is refused.

The control separates "game RL moved this instrument" from "RL at all moved it", and it is only worth
running if the match holds. A mismatched control is worse than no control, because its deltas would
still be reported beside the arms' and read as a statement about RL rather than about a differently
sized run, or one under a different loss. So the tests care most about `assert_compute_matched`
failing on each axis independently, and about the match being derived from what the reference arm
EXECUTED rather than what a plan asked for -- `games.train` sizes its micro-batch from the VRAM it
finds, so those two numbers differ by whatever the card decided.

Four of the eight axes are the loss itself: `beta`, `learning_rate`, `scale_rewards`, and the executed
estimator, which stands in for `loss_type` because it subsumes it. They are the ones with no symptom,
since a control matched on steps and episodes but minimising something else finishes, logs every
metric the end-of-run gate reads, and lands in the tables looking like a control. Two tests below
assert the live consequence rather than describing it: the arithmetic harness runs the arms' `beta`,
learning rate and estimator pair, so the launcher's own configuration passes the objective check
against the registered reference, and that same configuration set back to a hotter learning rate is
refused on that axis alone. Both go red if a harness default drifts off the arms' again, and the
second is what shows the first is a comparison rather than an axis nobody checks.

The estimator axis compares what EXECUTES, so two of its tests use pairs whose recorded `loss_type`
is identical and whose aggregation is not: one differing only in the fused-kernel flag, one only in
the micro-batch split under a Liger-unfaithful loss. A recorded-label check calls both a match.

A ninth thing is checked and is not an axis: which banked RUN of the reference arm the match came
from. Runs of the same arm at the same shape and objective are interchangeable on all eight axes, and
so are different arms of the same wave -- the banked 2B `twin-pd-self` record at the `twin-pd-group`
flagship's own micro-batch split reproduces it exactly -- so identity is the only thing that can tell
them apart, and without it a control's own record would carry a pairing nobody registered. The
launcher's wiring of that check is covered too, in `TestTheLauncherRunsBothGates`, because a gate
whose only caller can be deleted with the suite still green is a gate the next refactor drops.

The fixture is hand-written JSON, but every field name it uses is asserted against the real
dataclasses in `TestTheFixtureMatchesTheRealArtifact`. A hand-written fixture that drifted from the
artifact it stands in for would let this whole file pass while the reader broke.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from games import train_neutral
from games.arms import ARMS
from games.neutral_control import (
    ARITHMETIC_TASK,
    LEGACY_ESTIMATOR_VALUES,
    NEUTRAL_TASK_ARMS,
    RUN_CONFIG_FILENAME,
    ComputeMatch,
    NeutralTaskArm,
    ReferenceRun,
    assert_compute_matched,
    assert_registered_reference,
    compute_match_from_run_config,
    validate_neutral_task_arms,
)
from games.provenance import git_provenance
from games.sizing import SizingPlan
from games.train import RESUME_IDENTITY_DEFAULTS, GameTrainConfig
from games.train import run_config_payload as arm_run_config_payload
from games.train_neutral import matched_train_config, run_config_payload
from grpo.estimator_defaults import GRPO_LOSS_TYPE, GRPO_SCALE_REWARDS
from grpo.rlvr_math import TrainConfig as ArithmeticTrainConfig

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE_ARM = "twin-pd-group"
REFERENCE_MODEL = "Qwen/Qwen3.5-2B"

# The wave-1 shape: 70 optimizer steps of 8 prompts x group 8, which the card ran as 8 micro-batches
# of 8 episodes. Written as the artifact records it, not as the plan requested it.
STEPS = 70
GROUP = 8
MICRO_BATCH = 8
GRAD_ACCUM = 8
EPISODES_PER_STEP = MICRO_BATCH * GRAD_ACCUM

# The objective the game arms train under, asserted against `GameTrainConfig`'s own defaults below.
# The arithmetic control's harness reproduces both; they stay axes because they are that harness's own
# defaults rather than anything derived from the match, so either can drift off the arms' again.
REFERENCE_BETA = 0.0
REFERENCE_LEARNING_RATE = 1e-5

# Which run of the arm the fixture stands for. Any values would do; what matters is that a control
# registered against one run is refused a record of another.
REFERENCE_GIT_SHA = "3e0f5fd"
REFERENCE_STARTED_AT = "2026-08-21T18:32:51.277048+00:00"
REFERENCE_RUN = ReferenceRun(
    arm=REFERENCE_ARM, git_sha=REFERENCE_GIT_SHA, started_at=REFERENCE_STARTED_AT
)

# A KL strength no arm and no harness here runs, for the axis tests that need a live penalty to refuse.
# Written out rather than read off `ArithmeticTrainConfig`, which now agrees with the arms: taking it
# from there would leave those tests passing a matched value and asserting a refusal that never came.
A_LIVE_KL_PENALTY = 0.05
# A learning rate an order of magnitude above the arms', for the same reason: the harness trains at
# their 1e-5 now, so a refusal on this axis needs a mismatched value written out here.
A_HOTTER_LEARNING_RATE = 1e-4


def run_config(**overrides: Any) -> dict[str, Any]:
    """A reference arm's `run_config.json`, in the shape `games.train.run_config_payload` writes."""
    payload: dict[str, Any] = {
        "arm": REFERENCE_ARM,
        "game_id": "twin-pd",
        "grading": "group-mix",
        "git_sha": REFERENCE_GIT_SHA,
        "started_at": REFERENCE_STARTED_AT,
        "config": {
            "model_id": REFERENCE_MODEL,
            "max_steps": STEPS,
            "beta": REFERENCE_BETA,
            "learning_rate": REFERENCE_LEARNING_RATE,
            "loss_type": GRPO_LOSS_TYPE,
            "scale_rewards": GRPO_SCALE_REWARDS,
            "use_liger_kernel": True,
        },
        "sizing_plan": {
            "num_generations": GROUP,
            "micro_batch_size": MICRO_BATCH,
            "gradient_accumulation_steps": GRAD_ACCUM,
        },
    }
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        if field:
            payload[section][field] = value
        else:
            payload[section] = value
    return payload


def a_sizing_plan() -> SizingPlan:
    """A plan with no card behind it, for the tests that call the ARMS' own record writer.

    Built directly rather than through `games.sizing.plan_sizing`, which wants a model config and a
    VRAM figure to reason about: `games.train.run_config_payload` reads only `micro_batch_size` out of
    a plan and records the rest verbatim, so what a real card would have decided is not the subject.
    """
    return SizingPlan(
        num_generations=GROUP,
        prompts_per_step=GROUP,
        micro_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        episodes_per_step=EPISODES_PER_STEP,
        predicted_weight_gib=8.0,
        predicted_episode_gib=0.5,
        usable_vram_gib=44.0,
        episode_headroom_gib=36.0,
        episodes_that_fit=EPISODES_PER_STEP,
        clamped=False,
        reason="hand-built for a record-shape assertion, not sized from a card",
    )


def written(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / RUN_CONFIG_FILENAME
    path.write_text(json.dumps(payload))
    return path


def match(tmp_path: Path, **overrides: Any) -> ComputeMatch:
    return compute_match_from_run_config(
        written(tmp_path, run_config(**overrides)), reference_arm=REFERENCE_ARM
    )


def matched_kwargs(reference: ComputeMatch, **overrides: Any) -> dict[str, Any]:
    """Every argument of a control that reproduces the reference on all eight axes.

    `dict[str, Any]` deliberately, and the overrides merge in here rather than at the call site: the
    values span four types, so any narrower dict type cannot be splatted into a signature whose
    parameters are individually typed, and merging outside re-widens it.
    """
    return {
        "optimizer_steps": reference.optimizer_steps,
        "per_device_train_batch": MICRO_BATCH,
        "grad_accum_steps": GRAD_ACCUM,
        "num_generations": reference.num_generations,
        "model_id": reference.model_id,
        "beta": reference.beta,
        "learning_rate": reference.learning_rate,
        "loss_type": reference.loss_type,
        "use_liger_kernel": reference.use_liger_kernel,
        "scale_rewards": reference.scale_rewards,
    } | overrides


def reported_problems(refusal: ValueError) -> str:
    """Just the clause listing the axes that failed, out of the whole refusal message.

    Searching the full message cannot tell a reported axis from a mentioned one: the sentences after
    the list explain every axis by name and `ComputeMatch.describe` restates the reference's own
    learning rate and beta, so `"learning rate" in message` holds even when nothing checked it. That
    is not hypothetical -- disarming the learning-rate comparison left the whole-message form of this
    assertion green, and only this narrowing turned it red.
    """
    message = str(refusal)
    return message.split("not compute-matched: ", 1)[1].split(". matched to", 1)[0]


class TestTheProblemSlicerItself:
    """Cover the helper, because widening it back weakens every objective axis at once and silently.

    Each axis test below asserts a phrase is present in `reported_problems`. If that function ever
    returned the whole message again, four of those assertions would pass on prose instead of on a
    reported axis, and the four checks they stand for could all be disarmed with the suite still
    green. That is not a hypothetical: it is what the first draft of this file did.
    """

    def test_the_slice_drops_the_prose_that_names_axes_nothing_compared(
        self, tmp_path: Path
    ) -> None:
        reference = match(tmp_path)
        kwargs = matched_kwargs(reference, beta=A_LIVE_KL_PENALTY)
        with pytest.raises(ValueError, match="KL strength") as raised:
            assert_compute_matched(reference, **kwargs)
        whole = str(raised.value)
        sliced = reported_problems(raised.value)

        # Exactly one axis differs, so exactly one may be reported.
        assert "KL strength" in sliced
        # Each of these appears in the surrounding prose and in no reported problem. Asserting both
        # halves is what gives the test teeth: absent from the slice, present in the whole message.
        for mentioned_but_not_compared in ("learning rate", "scale_rewards", "loss_type"):
            assert mentioned_but_not_compared in whole
            assert mentioned_but_not_compared not in sliced
        assert "executed estimator" not in sliced

    def test_the_slice_keeps_every_axis_that_did_fail(self, tmp_path: Path) -> None:
        # The complement of the test above: narrowing must not drop a real problem either.
        reference = match(tmp_path)
        with pytest.raises(ValueError, match="not compute-matched") as raised:
            assert_compute_matched(
                reference,
                optimizer_steps=1,
                per_device_train_batch=1,
                grad_accum_steps=1,
                num_generations=2,
                model_id="Qwen/Qwen3.5-4B",
                beta=A_LIVE_KL_PENALTY,
                learning_rate=A_HOTTER_LEARNING_RATE,
                loss_type="dapo",
                scale_rewards="batch",
                use_liger_kernel=False,
            )
        sliced = reported_problems(raised.value)
        assert sliced.count(";") == 7  # eight axes, seven separators


class TestTheFixtureMatchesTheRealArtifact:
    """Without this, a rename upstream would leave every test below green and the reader broken."""

    def test_the_sizing_fields_read_exist_on_the_real_plan(self) -> None:
        fields = {field.name for field in dataclasses.fields(SizingPlan)}
        assert {"num_generations", "micro_batch_size", "gradient_accumulation_steps"} <= fields

    def test_the_config_fields_read_exist_on_the_real_training_config(self) -> None:
        fields = {field.name for field in dataclasses.fields(GameTrainConfig)}
        assert {
            "model_id",
            "max_steps",
            "beta",
            "learning_rate",
            "loss_type",
            "scale_rewards",
            "use_liger_kernel",
        } <= fields

    def test_the_fixtures_objective_is_the_one_the_game_arms_actually_train_under(self) -> None:
        """A fixture holding an objective no arm runs would make every refusal below meaningless.

        Read off the field defaults rather than a built config: `GameTrainConfig.__post_init__`
        demands a corpus and a live vLLM rollout path, neither of which this assertion is about.
        """
        defaults = {field.name: field.default for field in dataclasses.fields(GameTrainConfig)}
        assert defaults["beta"] == REFERENCE_BETA
        assert defaults["learning_rate"] == REFERENCE_LEARNING_RATE
        assert defaults["loss_type"] == GRPO_LOSS_TYPE
        assert defaults["scale_rewards"] == GRPO_SCALE_REWARDS

    def test_the_identity_fields_read_are_the_ones_the_real_writer_records(self) -> None:
        """A renamed key on the writer would leave the identity check reading None on every record.

        Pinned against `games.train.run_config_payload`, which writes every record the check reads.
        Pinning it against the control's own sibling writer instead looks equivalent and is not: the
        two payloads share a convention rather than a code path, so the arms' key could be renamed
        with the control's untouched and this assertion would stay green. Watched: renaming
        `started_at` in `games/train.py` leaves the whole scoped suite passing without this form.
        """
        payload = arm_run_config_payload(
            GameTrainConfig(arm=REFERENCE_ARM, generate_fresh=True),
            plan=a_sizing_plan(),
            device={},
            derived={},
        )
        assert {"arm", "git_sha", "started_at"} <= payload.keys()
        # Separately, because `git_provenance` could rename its own key while `games.train` goes on
        # splatting whatever it returns into the record.
        assert "git_sha" in git_provenance()

    def test_the_arithmetic_harness_agrees_with_the_arms_on_every_objective_axis(self) -> None:
        """What lets `train_neutral` launch as configured, pinned against the arms' own defaults.

        `beta` and the learning rate are `grpo.rlvr_math`'s own defaults and the estimator pair is a
        shared constant, so nothing but the launch-time check ties the harness to the arms -- and this
        is the same statement made where a test can see it. It goes red the moment either default
        drifts, which is also the moment the launcher starts refusing; the refusal is the right
        outcome then, and this test is the one that says which knob moved.
        """
        arms = {field.name: field.default for field in dataclasses.fields(GameTrainConfig)}
        harness = ArithmeticTrainConfig()
        assert harness.beta == arms["beta"] == REFERENCE_BETA
        assert harness.learning_rate == arms["learning_rate"] == REFERENCE_LEARNING_RATE

    def test_the_legacy_estimator_values_agree_with_the_trainers_own_mapping(self) -> None:
        """The restated copy has to say what `games.train` says, or a resume and a match disagree.

        `games.neutral_control` cannot import the mapping, because `games.train` imports torch and
        that module is deliberately importable without a training stack. So the copy is pinned here.
        """
        assert {
            field: RESUME_IDENTITY_DEFAULTS[field] for field in LEGACY_ESTIMATOR_VALUES
        } == LEGACY_ESTIMATOR_VALUES


class TestTheRegistry:
    def test_the_planned_controls_are_all_registered(self) -> None:
        assert set(NEUTRAL_TASK_ARMS) == {"neutral-arithmetic"}

    def test_every_control_is_matched_to_a_registered_game_arm(self) -> None:
        for name, arm in NEUTRAL_TASK_ARMS.items():
            assert arm.matched_to in ARMS, name
            assert arm.notes, name

    def test_the_arithmetic_control_is_matched_to_the_headline_arm(self) -> None:
        # Matched to the arm every instrument delta is currently read against, so a shift here
        # relocates the baseline for the whole battery rather than for one arm.
        assert NEUTRAL_TASK_ARMS["neutral-arithmetic"].matched_to == "twin-pd-group"
        assert NEUTRAL_TASK_ARMS["neutral-arithmetic"].task == ARITHMETIC_TASK

    def test_the_arithmetic_control_names_which_run_of_that_arm_it_is_matched_to(self) -> None:
        """The arm name alone does not identify a run, and this arm has been run many times."""
        registered = NEUTRAL_TASK_ARMS["neutral-arithmetic"].reference_run
        assert registered.arm == "twin-pd-group"
        assert registered.git_sha
        assert registered.started_at

    def test_a_control_matched_to_nothing_real_is_refused(self) -> None:
        arms = {
            "bad": NeutralTaskArm(
                task=ARITHMETIC_TASK,
                matched_to="twin-pd-vibes",
                reference_run=dataclasses.replace(REFERENCE_RUN, arm="twin-pd-vibes"),
                notes="n/a",
            )
        }
        with pytest.raises(ValueError, match=re.escape("not in games.arms.ARMS")):
            validate_neutral_task_arms(arms)

    def test_a_control_sharing_a_name_with_a_game_arm_is_refused(self) -> None:
        # Every readout resolves an arm label through ARMS, so the two runs' artifacts would merge
        # into one row and neither could be read.
        arms = {
            REFERENCE_ARM: NeutralTaskArm(
                task=ARITHMETIC_TASK,
                matched_to=REFERENCE_ARM,
                reference_run=REFERENCE_RUN,
                notes="n/a",
            )
        }
        with pytest.raises(ValueError, match="shares a name with a registered game arm"):
            validate_neutral_task_arms(arms)

    def test_a_reference_run_of_a_different_arm_than_matched_to_is_refused(self) -> None:
        """Otherwise the launcher derives the match from one arm and labels the control the other."""
        arms = {
            "bad": NeutralTaskArm(
                task=ARITHMETIC_TASK,
                matched_to=REFERENCE_ARM,
                reference_run=dataclasses.replace(REFERENCE_RUN, arm="twin-pd-self"),
                notes="n/a",
            )
        }
        with pytest.raises(ValueError, match="name different"):
            validate_neutral_task_arms(arms)

    @pytest.mark.parametrize(
        "unidentified",
        [
            dataclasses.replace(REFERENCE_RUN, git_sha=""),
            dataclasses.replace(REFERENCE_RUN, started_at=""),
        ],
    )
    def test_a_reference_run_that_does_not_identify_itself_is_refused(
        self, unidentified: ReferenceRun
    ) -> None:
        # With either field empty, any record of the right arm satisfies the reference check, which
        # is the whole thing the check exists to prevent.
        arms = {
            "bad": NeutralTaskArm(
                task=ARITHMETIC_TASK,
                matched_to=REFERENCE_ARM,
                reference_run=unidentified,
                notes="n/a",
            )
        }
        with pytest.raises(ValueError, match="does not identify itself"):
            validate_neutral_task_arms(arms)

    def test_an_undescribed_control_is_refused(self) -> None:
        arms = {
            "bad": NeutralTaskArm(
                task=ARITHMETIC_TASK,
                matched_to=REFERENCE_ARM,
                reference_run=REFERENCE_RUN,
                notes="",
            )
        }
        with pytest.raises(ValueError, match="UNREGISTERED"):
            validate_neutral_task_arms(arms)

    def test_the_live_registry_passes(self) -> None:
        validate_neutral_task_arms(NEUTRAL_TASK_ARMS)


class TestDerivingTheMatch:
    def test_the_shape_comes_from_the_sizing_plan_not_the_request(self, tmp_path: Path) -> None:
        derived = match(tmp_path)
        assert derived.optimizer_steps == STEPS
        assert derived.episodes_per_step == EPISODES_PER_STEP
        assert derived.num_generations == GROUP
        assert derived.prompts_per_step == EPISODES_PER_STEP // GROUP
        assert derived.total_episodes == STEPS * EPISODES_PER_STEP
        assert derived.model_id == REFERENCE_MODEL

    def test_the_objective_comes_from_the_config_section(self, tmp_path: Path) -> None:
        derived = match(tmp_path)
        assert derived.beta == REFERENCE_BETA
        assert derived.learning_rate == REFERENCE_LEARNING_RATE
        assert derived.loss_type == GRPO_LOSS_TYPE
        assert derived.scale_rewards == GRPO_SCALE_REWARDS

    def test_the_description_names_both_axes_the_model_and_the_objective(
        self, tmp_path: Path
    ) -> None:
        described = match(tmp_path).describe()
        assert f"{STEPS} optimizer steps" in described
        assert f"{EPISODES_PER_STEP} episodes/step" in described
        assert REFERENCE_MODEL in described
        # Logged at every launch, so the run's own log says what objective it claims to match. The
        # estimator appears as the aggregation that executed rather than as the recorded label.
        assert f"executed estimator '{GRPO_LOSS_TYPE} (faithful under Liger)'" in described
        assert f"scale_rewards {GRPO_SCALE_REWARDS!r}" in described
        assert f"beta {REFERENCE_BETA}" in described

    def test_the_identity_comes_from_the_records_own_provenance(self, tmp_path: Path) -> None:
        derived = match(tmp_path)
        assert derived.run_identity == REFERENCE_RUN
        # In the launch log beside the shape, so a run's own log says which banked run it matched.
        assert REFERENCE_GIT_SHA in derived.describe()
        assert REFERENCE_STARTED_AT in derived.describe()

    def test_a_record_of_a_different_arm_is_refused_before_any_axis_is_compared(
        self, tmp_path: Path
    ) -> None:
        """The live hazard: same wave, same shape, same objective, different experiment.

        The banked 2B `twin-pd-self` record at the flagship's own micro-batch split reproduces its 70
        steps, 64 episodes per step, group 8, model and objective, so the axes cannot tell the two
        records apart and the only trace of a substitution is the label this module would print. (The
        two self records banked at a different split are caught on the executed-estimator axis, which
        is why this is about identity rather than about the axes being weak.)
        """
        with pytest.raises(ValueError, match=re.escape("is a run of 'twin-pd-self'")):
            match(tmp_path, arm="twin-pd-self")

    @pytest.mark.parametrize("bad", ["", None, 3])
    def test_a_record_naming_no_arm_is_refused(self, tmp_path: Path, bad: object) -> None:
        with pytest.raises(ValueError, match="which experiment produced it is unknown"):
            match(tmp_path, arm=bad)

    @pytest.mark.parametrize("field", ["git_sha", "started_at"])
    @pytest.mark.parametrize("how", ["absent", "empty"])
    def test_a_record_that_does_not_identify_its_run_is_refused(
        self, tmp_path: Path, field: str, how: str
    ) -> None:
        # Both are written by games.train on every launch, so an absence is a record this module
        # cannot check against a registration rather than an older convention to accommodate. An
        # empty string is checked as well as an absent key: it would satisfy an isinstance-only read
        # and then match any other record whose same field is empty.
        payload = run_config()
        if how == "absent":
            del payload[field]
        else:
            payload[field] = ""
        with pytest.raises(ValueError, match="does not identify which run"):
            compute_match_from_run_config(written(tmp_path, payload), reference_arm=REFERENCE_ARM)

    def test_a_record_written_before_the_estimator_fields_reads_as_what_it_ran(
        self, tmp_path: Path
    ) -> None:
        """Every banked twin-pd-group arm predates those fields, so this is the live path today."""
        payload = run_config()
        del payload["config"]["loss_type"]
        del payload["config"]["scale_rewards"]
        derived = compute_match_from_run_config(
            written(tmp_path, payload), reference_arm=REFERENCE_ARM
        )
        assert derived.loss_type == LEGACY_ESTIMATOR_VALUES["loss_type"]
        assert derived.scale_rewards == LEGACY_ESTIMATOR_VALUES["scale_rewards"]

    def test_a_record_with_no_kl_strength_is_refused_rather_than_guessed(
        self, tmp_path: Path
    ) -> None:
        # `beta` has been a GameTrainConfig field since the games package was added, so its absence
        # means this is not a games run config, and there is no earlier value to read it as.
        payload = run_config()
        del payload["config"]["beta"]
        with pytest.raises(ValueError, match=re.escape("has no config.beta")):
            compute_match_from_run_config(written(tmp_path, payload), reference_arm=REFERENCE_ARM)

    def test_a_record_with_no_learning_rate_is_refused(self, tmp_path: Path) -> None:
        payload = run_config()
        del payload["config"]["learning_rate"]
        with pytest.raises(ValueError, match=re.escape("has no config.learning_rate")):
            compute_match_from_run_config(written(tmp_path, payload), reference_arm=REFERENCE_ARM)

    @pytest.mark.parametrize("bad", ["1e-5", None, True, float("nan"), float("inf")])
    def test_a_learning_rate_that_is_not_a_finite_number_is_refused(
        self, tmp_path: Path, bad: object
    ) -> None:
        with pytest.raises(ValueError, match="not a finite number"):
            match(tmp_path, **{"config.learning_rate": bad})

    def test_a_negative_kl_strength_condemns_the_whole_record(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a run that happened"):
            match(tmp_path, **{"config.beta": -0.05})

    def test_a_record_with_no_fused_kernel_flag_is_refused(self, tmp_path: Path) -> None:
        # One of the three inputs to the executed aggregation, and recorded by all 29 banked games
        # runs, so its absence is a record this module cannot read rather than an older convention.
        payload = run_config()
        del payload["config"]["use_liger_kernel"]
        with pytest.raises(ValueError, match=re.escape("has no config.use_liger_kernel")):
            compute_match_from_run_config(written(tmp_path, payload), reference_arm=REFERENCE_ARM)

    @pytest.mark.parametrize("bad", ["true", 1, None])
    def test_a_fused_kernel_flag_that_is_not_a_boolean_is_refused(
        self, tmp_path: Path, bad: object
    ) -> None:
        with pytest.raises(ValueError, match="is not a boolean"):
            match(tmp_path, **{"config.use_liger_kernel": bad})

    def test_a_loss_type_trl_does_not_accept_is_refused(self, tmp_path: Path) -> None:
        # Checked here rather than left to the comparison, because an unknown label has no executed
        # aggregation to derive and `executed_estimator` would raise naming no artifact.
        with pytest.raises(ValueError, match="which TRL does not accept"):
            match(tmp_path, **{"config.loss_type": "dr_grp"})

    def test_a_reward_scaling_trl_does_not_accept_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="which TRL does not accept"):
            match(tmp_path, **{"config.scale_rewards": "grouped"})

    @pytest.mark.parametrize("bad", ["", None, 3])
    def test_an_estimator_field_naming_no_estimator_is_refused(
        self, tmp_path: Path, bad: object
    ) -> None:
        # Absence reads as the legacy value; a field that is present and unusable does not.
        with pytest.raises(ValueError, match="names no estimator"):
            match(tmp_path, **{"config.loss_type": bad})

    def test_an_unregistered_reference_arm_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=re.escape("not in games.arms.ARMS")):
            compute_match_from_run_config(
                written(tmp_path, run_config()), reference_arm="twin-pd-vibes"
            )

    def test_a_missing_run_config_explains_that_the_arm_has_to_have_run(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="until that arm has run"):
            compute_match_from_run_config(tmp_path / "absent.json", reference_arm=REFERENCE_ARM)

    def test_a_run_config_missing_the_sizing_plan_is_refused(self, tmp_path: Path) -> None:
        payload = run_config()
        del payload["sizing_plan"]
        with pytest.raises(ValueError, match=re.escape("sizing_plan.micro_batch_size")):
            compute_match_from_run_config(written(tmp_path, payload), reference_arm=REFERENCE_ARM)

    def test_a_run_config_recording_no_model_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="shares no instrument baseline"):
            match(tmp_path, **{"config.model_id": None})

    def test_an_episode_count_the_group_does_not_divide_is_refused(self, tmp_path: Path) -> None:
        # TRL's sampler emits whole groups, so this shape cannot have run: the artifact disagrees
        # with itself and nothing derived from it is trustworthy.
        with pytest.raises(ValueError, match="does not divide it"):
            match(
                tmp_path,
                **{"sizing_plan.micro_batch_size": 3, "sizing_plan.gradient_accumulation_steps": 1},
            )

    @pytest.mark.parametrize("bad", [0, -4, True, "eight", None])
    def test_a_step_count_that_is_not_a_positive_integer_is_refused(
        self, tmp_path: Path, bad: object
    ) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            match(tmp_path, **{"config.max_steps": bad})


class TestTheMatchIsCheckedOnEveryAxis:
    """Each axis alone is satisfiable while the comparison is meaningless, so all eight are checked."""

    @pytest.fixture
    def reference(self, tmp_path: Path) -> ComputeMatch:
        return match(tmp_path)

    def test_the_matched_configuration_passes(self, reference: ComputeMatch) -> None:
        assert_compute_matched(reference, **matched_kwargs(reference))

    def test_a_different_micro_batch_passes_when_the_product_is_the_same(
        self, reference: ComputeMatch
    ) -> None:
        # Splitting the same episodes across more micro-batches is a memory decision, not an
        # experimental one, so the check is on the product rather than on either factor.
        kwargs = matched_kwargs(reference)
        kwargs["per_device_train_batch"] = MICRO_BATCH * 2
        kwargs["grad_accum_steps"] = GRAD_ACCUM // 2
        assert_compute_matched(reference, **kwargs)

    def test_fewer_steps_is_refused(self, reference: ComputeMatch) -> None:
        kwargs = matched_kwargs(reference)
        kwargs["optimizer_steps"] = STEPS // 2
        with pytest.raises(ValueError, match="optimizer steps"):
            assert_compute_matched(reference, **kwargs)

    def test_fewer_episodes_per_step_is_refused(self, reference: ComputeMatch) -> None:
        kwargs = matched_kwargs(reference)
        kwargs["grad_accum_steps"] = 1
        with pytest.raises(ValueError, match="episodes per step"):
            assert_compute_matched(reference, **kwargs)

    def test_a_different_group_size_is_refused(self, reference: ComputeMatch) -> None:
        kwargs = matched_kwargs(reference)
        kwargs["num_generations"] = GROUP // 2
        with pytest.raises(ValueError, match="group size"):
            assert_compute_matched(reference, **kwargs)

    def test_a_different_base_model_is_refused(self, reference: ComputeMatch) -> None:
        kwargs = matched_kwargs(reference)
        kwargs["model_id"] = "Qwen/Qwen3.5-4B"
        with pytest.raises(ValueError, match="base model"):
            assert_compute_matched(reference, **kwargs)

    def test_a_live_kl_penalty_against_an_arm_without_one_is_refused(
        self, reference: ComputeMatch
    ) -> None:
        """No harness here runs a KL anchor now, so this axis guards a future arm rather than today's.

        Kept at full strength for that reason: `beta` is the objective knob whose mismatch changes the
        loss's shape rather than its scale, and the one a control could pick up again from a copied
        recipe without anything downstream showing it.
        """
        kwargs = matched_kwargs(reference)
        kwargs["beta"] = A_LIVE_KL_PENALTY
        with pytest.raises(ValueError, match="KL strength") as raised:
            assert_compute_matched(reference, **kwargs)
        reported = reported_problems(raised.value)
        assert f"beta={A_LIVE_KL_PENALTY}" in reported
        assert f"the reference's {REFERENCE_BETA}" in reported

    def test_a_different_learning_rate_is_refused(self, reference: ComputeMatch) -> None:
        kwargs = matched_kwargs(reference)
        kwargs["learning_rate"] = A_HOTTER_LEARNING_RATE
        with pytest.raises(ValueError, match="learning rate") as raised:
            assert_compute_matched(reference, **kwargs)
        reported = reported_problems(raised.value)
        assert f"learning rate {A_HOTTER_LEARNING_RATE}" in reported
        assert f"the reference's {REFERENCE_LEARNING_RATE}" in reported

    def test_a_different_loss_type_is_refused(self, reference: ComputeMatch) -> None:
        kwargs = matched_kwargs(reference)
        kwargs["loss_type"] = "dapo"
        with pytest.raises(ValueError, match="executed estimator") as raised:
            assert_compute_matched(reference, **kwargs)
        reported = reported_problems(raised.value)
        # Named by mechanism rather than by label, and the label cannot hide, because every branch
        # of `executed_estimator` opens with the loss type's own name. The fallback branch is the
        # per-micro-batch one because this fixture's micro-batch is 8; at 1 it would read
        # per-sequence, which is the distinction the sibling test below turns on.
        assert "dapo -> per-micro-batch token normalization via Liger fallback" in reported
        assert f"the reference's '{GRPO_LOSS_TYPE} (faithful under Liger)'" in reported

    def test_the_same_recorded_loss_type_off_the_fused_kernel_is_refused(
        self, reference: ComputeMatch
    ) -> None:
        """The case a recorded-label check cannot see: identical in every recorded field but one flag.

        TRL's Liger call site does not forward `num_items_in_batch`, so the fused path aggregates
        differently from the plain one. A control that records the same `loss_type` as its reference
        and merely runs the other kernel optimises a different objective, and comparing labels reports
        a match.
        """
        legacy = dataclasses.replace(reference, loss_type="dapo")
        kwargs = matched_kwargs(legacy)
        kwargs["loss_type"] = "dapo"
        kwargs["use_liger_kernel"] = False
        with pytest.raises(ValueError, match="executed estimator") as raised:
            assert_compute_matched(legacy, **kwargs)
        reported = reported_problems(raised.value)
        assert "dapo (non-Liger TRL path)" in reported
        assert "dapo -> per-micro-batch token normalization via Liger fallback" in reported
        # The recorded label is identical on both sides, so nothing else may be reported.
        assert "scale_rewards" not in reported
        assert "KL strength" not in reported

    def test_the_micro_batch_split_is_refused_under_a_liger_unfaithful_loss(
        self, reference: ComputeMatch
    ) -> None:
        """The other half of the product rule, which the product alone cannot express.

        Its sibling above pins the free case: splitting the same episodes across more micro-batches
        is a memory decision under `dr_grpo`, whose executed aggregation does not mention the
        micro-batch. Under a Liger-unfaithful loss the micro-batch decides the normalizer, so the
        identical split stops being free and becomes an experimental change.
        """
        legacy = dataclasses.replace(reference, loss_type="dapo", micro_batch_size=MICRO_BATCH)
        split = matched_kwargs(legacy)
        split["loss_type"] = "dapo"
        split["per_device_train_batch"] = 1
        split["grad_accum_steps"] = EPISODES_PER_STEP
        with pytest.raises(ValueError, match="executed estimator") as raised:
            assert_compute_matched(legacy, **split)
        reported = reported_problems(raised.value)
        assert "per-sequence GRPO via Liger fallback" in reported
        assert "per-micro-batch token normalization via Liger fallback" in reported
        assert "episodes per step" not in reported

    def test_a_different_reward_scaling_is_refused(self, reference: ComputeMatch) -> None:
        kwargs = matched_kwargs(reference)
        kwargs["scale_rewards"] = "batch"
        with pytest.raises(ValueError, match="scale_rewards") as raised:
            assert_compute_matched(reference, **kwargs)
        reported = reported_problems(raised.value)
        assert "scale_rewards 'batch'" in reported
        assert f"the reference's {GRPO_SCALE_REWARDS!r}" in reported

    def test_an_objective_mismatch_says_where_the_controls_own_values_come_from(
        self, reference: ComputeMatch
    ) -> None:
        """A launch log has to be readable without this module beside it."""
        kwargs = matched_kwargs(reference)
        kwargs["learning_rate"] = A_HOTTER_LEARNING_RATE
        with pytest.raises(ValueError, match=re.escape("grpo.rlvr_math.TrainConfig")):
            assert_compute_matched(reference, **kwargs)

    def test_a_size_only_mismatch_does_not_lecture_about_the_objective(
        self, reference: ComputeMatch
    ) -> None:
        kwargs = matched_kwargs(reference)
        kwargs["optimizer_steps"] = STEPS // 2
        with pytest.raises(ValueError, match="optimizer steps") as raised:
            assert_compute_matched(reference, **kwargs)
        assert "grpo.rlvr_math.TrainConfig" not in str(raised.value)

    def test_every_mismatched_axis_is_reported_at_once(self, reference: ComputeMatch) -> None:
        # One raise per launch, so listing them all is the difference between one round trip and eight.
        with pytest.raises(ValueError, match="not compute-matched") as raised:
            assert_compute_matched(
                reference,
                optimizer_steps=1,
                per_device_train_batch=1,
                grad_accum_steps=1,
                num_generations=2,
                model_id="Qwen/Qwen3.5-4B",
                beta=A_LIVE_KL_PENALTY,
                learning_rate=A_HOTTER_LEARNING_RATE,
                loss_type="dapo",
                scale_rewards="batch",
                use_liger_kernel=False,
            )
        reported = reported_problems(raised.value)
        for axis in (
            "optimizer steps",
            "episodes per step",
            "group size",
            "base model",
            "KL strength",
            "learning rate",
            "executed estimator",
            "scale_rewards",
        ):
            assert axis in reported


class TestTheReferenceIsTheRegisteredRun:
    """The ninth check, and the only one the eight axes cannot stand in for."""

    @pytest.fixture
    def reference(self, tmp_path: Path) -> ComputeMatch:
        return match(tmp_path)

    def test_the_registered_run_passes(self, reference: ComputeMatch) -> None:
        assert_registered_reference(reference, REFERENCE_RUN)

    @pytest.mark.parametrize(
        "other_run",
        [
            dataclasses.replace(REFERENCE_RUN, git_sha="0000000"),
            dataclasses.replace(REFERENCE_RUN, started_at="2026-08-19T08:21:57.626292+00:00"),
        ],
    )
    def test_another_run_of_the_same_arm_is_refused(
        self, reference: ComputeMatch, other_run: ReferenceRun
    ) -> None:
        """A re-run at the same shape and objective is identical on all eight axes, and is not it."""
        with pytest.raises(ValueError, match="registered against") as raised:
            assert_registered_reference(reference, other_run)
        message = str(raised.value)
        # Both runs named, so the operator can see which one to keep without opening either file.
        assert reference.run_identity.describe() in message
        assert other_run.describe() in message

    def test_the_refusal_names_the_sha_spelling_trap(self, reference: ComputeMatch) -> None:
        """The comparison is byte-exact and the two producers spell a sha differently.

        `games.provenance.git_sha` returns the short `GIT_SHA` a container baked in, or the forty
        characters `git rev-parse HEAD` prints on a box with a checkout, so one commit has two
        spellings and they refuse each other. An operator who registers the short form and then re-runs
        the reference locally reads "a different run" for the same commit, and the cheap way out is the
        one thing this refusal tells them not to do -- so the refusal has to say the timestamps are
        where to look.
        """
        long_form = dataclasses.replace(REFERENCE_RUN, git_sha=REFERENCE_GIT_SHA + "0" * 33)
        with pytest.raises(ValueError, match="Read the timestamps") as raised:
            assert_registered_reference(reference, long_form)
        assert "spelling" in str(raised.value)

    def test_the_refusal_says_how_to_adopt_a_superseding_run(self, reference: ComputeMatch) -> None:
        # Otherwise the cheap way past it is to keep pointing --reference-run-config elsewhere, which
        # leaves the registry saying one thing and every run doing another.
        with pytest.raises(ValueError, match="updating reference_run"):
            assert_registered_reference(
                reference, dataclasses.replace(REFERENCE_RUN, git_sha="0000000")
            )


class TestTheLauncherBuildsAMatchedConfig:
    @pytest.fixture
    def reference(self, tmp_path: Path) -> ComputeMatch:
        return match(tmp_path)

    def test_the_config_reproduces_the_reference_shape(
        self, reference: ComputeMatch, tmp_path: Path
    ) -> None:
        config = matched_train_config(
            reference,
            output_dir=tmp_path,
            run_name="neutral-arithmetic",
            grad_accum_steps=GRAD_ACCUM,
        )
        assert config.max_steps_full == STEPS
        assert config.per_device_train_batch * config.grad_accum_steps == EPISODES_PER_STEP
        assert config.num_generations == GROUP
        assert config.model_id == REFERENCE_MODEL
        # The reference's own objective, so this asserts the shape axes and nothing else; what the
        # launcher's real objective does against them is the test below.
        assert_compute_matched(
            reference,
            optimizer_steps=config.max_steps_full,
            per_device_train_batch=config.per_device_train_batch,
            grad_accum_steps=config.grad_accum_steps,
            num_generations=config.num_generations,
            model_id=config.model_id,
            beta=reference.beta,
            learning_rate=reference.learning_rate,
            loss_type=reference.loss_type,
            scale_rewards=reference.scale_rewards,
            use_liger_kernel=reference.use_liger_kernel,
        )

    def check_the_launchers_own_objective(
        self, reference: ComputeMatch, config: ArithmeticTrainConfig
    ) -> None:
        """The call `train_neutral.main` makes, argument for argument.

        The shape comes from the config the launcher built, `beta` and the learning rate from the
        arithmetic harness's own defaults riding in that config, and the estimator pair from the
        repo-wide constants, because that is where `grpo.rlvr_math` reads them.
        """
        assert_compute_matched(
            reference,
            optimizer_steps=config.max_steps_full,
            per_device_train_batch=config.per_device_train_batch,
            grad_accum_steps=config.grad_accum_steps,
            num_generations=config.num_generations,
            model_id=config.model_id,
            beta=config.beta,
            learning_rate=config.learning_rate,
            loss_type=GRPO_LOSS_TYPE,
            scale_rewards=GRPO_SCALE_REWARDS,
            use_liger_kernel=config.use_liger_kernel,
        )

    def test_the_launchers_own_objective_passes_against_the_registered_reference(
        self, reference: ComputeMatch, tmp_path: Path
    ) -> None:
        """The state of the world, asserted rather than described in a note.

        `matched_train_config` reproduces the reference arm's shape and leaves the arithmetic
        harness's own loss alone, and that loss agrees with the arms on every objective axis: `beta`,
        the learning rate, and an estimator pair both sides read from the same constants. So the
        launcher's preflight passes against the registered reference and is refused on nothing, which
        is what lets the control run at all. A failure here reads as "a harness default drifted off the
        arms'"; the sibling below is what shows this pass is a comparison rather than a disarmed axis.
        """
        config = matched_train_config(
            reference,
            output_dir=tmp_path,
            run_name="neutral-arithmetic",
            grad_accum_steps=GRAD_ACCUM,
        )
        self.check_the_launchers_own_objective(reference, config)

    def test_a_harness_set_back_to_a_hotter_learning_rate_is_refused_on_that_axis_alone(
        self, reference: ComputeMatch, tmp_path: Path
    ) -> None:
        """The negative that keeps the pass above honest: the learning-rate axis is live.

        The same call the launcher makes, with only the harness's learning rate moved off the arms'.
        The refusal has to name that axis and nothing else: a second axis appearing here would mean
        the pass above was hiding one, and the learning rate going unreported would mean the axis had
        been disarmed and the pass was vacuous.
        """
        config = dataclasses.replace(
            matched_train_config(
                reference,
                output_dir=tmp_path,
                run_name="neutral-arithmetic",
                grad_accum_steps=GRAD_ACCUM,
            ),
            learning_rate=A_HOTTER_LEARNING_RATE,
        )
        with pytest.raises(ValueError, match="not compute-matched") as raised:
            self.check_the_launchers_own_objective(reference, config)
        reported = reported_problems(raised.value)
        assert f"learning rate {A_HOTTER_LEARNING_RATE}" in reported
        assert f"the reference's {REFERENCE_LEARNING_RATE}" in reported
        assert "KL strength" not in reported
        # Both sides read the estimator from grpo.estimator_defaults, run the fused kernel and take
        # the micro-batch from the same shape, so the executed aggregation matches by construction.
        assert "executed estimator" not in reported
        assert "scale_rewards" not in reported
        for size_axis in ("optimizer steps", "episodes per step", "group size", "base model"):
            assert size_axis not in reported
        # One axis, so no separator: the refusal names the learning rate and nothing else.
        assert ";" not in reported

    def test_the_quick_path_is_off_so_the_matched_step_count_is_the_one_used(
        self, reference: ComputeMatch, tmp_path: Path
    ) -> None:
        # `quick_run=True` would take `max_steps_quick` and silently ignore the one field whose whole
        # purpose here is to be matched.
        config = matched_train_config(
            reference, output_dir=tmp_path, run_name="n", grad_accum_steps=1
        )
        assert config.quick_run is False

    def test_an_accumulation_that_does_not_divide_the_episodes_is_refused(
        self, reference: ComputeMatch, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="does not divide"):
            matched_train_config(reference, output_dir=tmp_path, run_name="n", grad_accum_steps=3)

    def test_the_run_config_records_the_pairing_and_the_executed_estimator(
        self, reference: ComputeMatch, tmp_path: Path
    ) -> None:
        config = matched_train_config(
            reference, output_dir=tmp_path, run_name="neutral-arithmetic", grad_accum_steps=1
        )
        payload = run_config_payload(
            arm="neutral-arithmetic", config=config, match=reference, task=ARITHMETIC_TASK
        )
        assert payload["arm"] == "neutral-arithmetic"
        assert payload["matched_to"] == REFERENCE_ARM
        assert payload["task"] == ARITHMETIC_TASK
        assert payload["config"]["model_id"] == REFERENCE_MODEL
        assert payload["executed_estimator"]
        # None rather than a fabricated game: the readouts brand this run UNREGISTERED, which is
        # true, and inventing a game_id to quiet the banner would make it false.
        assert payload["game_id"] is None
        assert payload["grading"] is None

    def test_the_run_config_records_the_objective_the_control_trains_under(
        self, reference: ComputeMatch, tmp_path: Path
    ) -> None:
        """A knob no artifact records is a knob nobody auditing the run afterwards can see.

        `beta` and the learning rate ride in through the config dataclass; the estimator pair has no
        field on the arithmetic harness at all, which is why the payload carries it at the top level.
        """
        config = matched_train_config(
            reference, output_dir=tmp_path, run_name="neutral-arithmetic", grad_accum_steps=1
        )
        payload = run_config_payload(
            arm="neutral-arithmetic", config=config, match=reference, task=ARITHMETIC_TASK
        )
        assert payload["config"]["beta"] == ArithmeticTrainConfig().beta
        # The value itself, not only the round trip: a config-level revert to a KL-anchored control
        # would satisfy the line above and leave every artifact of the run reading as anchored.
        assert payload["config"]["beta"] == 0.0
        assert payload["config"]["learning_rate"] == ArithmeticTrainConfig().learning_rate
        # And the learning rate's value for the same reason: a harness drifted back to a hotter rate
        # would round-trip into this record just as cleanly, and this is the line that would say so.
        assert payload["config"]["learning_rate"] == REFERENCE_LEARNING_RATE
        assert payload["loss_type"] == GRPO_LOSS_TYPE
        assert payload["scale_rewards"] == GRPO_SCALE_REWARDS
        # And the reference's side of each axis, so one record holds both halves of the comparison.
        assert payload["compute_match"]["beta"] == REFERENCE_BETA
        assert payload["compute_match"]["learning_rate"] == REFERENCE_LEARNING_RATE
        assert payload["compute_match"]["loss_type"] == GRPO_LOSS_TYPE
        assert payload["compute_match"]["scale_rewards"] == GRPO_SCALE_REWARDS

    def test_the_run_config_records_which_banked_run_the_compute_was_matched_to(
        self, reference: ComputeMatch, tmp_path: Path
    ) -> None:
        """The pairing is the whole artifact of a control run, so the record has to carry its half."""
        config = matched_train_config(
            reference, output_dir=tmp_path, run_name="neutral-arithmetic", grad_accum_steps=1
        )
        payload = run_config_payload(
            arm="neutral-arithmetic", config=config, match=reference, task=ARITHMETIC_TASK
        )
        assert payload["compute_match"]["run_identity"] == dataclasses.asdict(REFERENCE_RUN)

    def test_the_run_config_is_json_serialisable(
        self, reference: ComputeMatch, tmp_path: Path
    ) -> None:
        # `TrainConfig.dtype` is a torch object and would take the whole payload down with it.
        config = matched_train_config(
            reference, output_dir=tmp_path, run_name="n", grad_accum_steps=1
        )
        payload = run_config_payload(
            arm="neutral-arithmetic", config=config, match=reference, task=ARITHMETIC_TASK
        )
        assert json.loads(json.dumps(payload))["matched_to"] == REFERENCE_ARM


class TestTheLauncherRunsBothGates:
    """Drive `games.train_neutral.main` itself, because a gate nothing calls is a gate nothing runs.

    Every other test in this file calls the checks directly, which covers what each one refuses and
    not whether a launch reaches it. That gap was measured rather than imagined: deleting
    `assert_registered_reference(match, arm.reference_run)` out of the launcher left the whole scoped
    suite green, and while the harness's learning rate still stood off the arms' the identity gate
    could not be reached on a real launch either, so an operator would not have noticed. The same
    hole covered `assert_compute_matched`'s wiring and the ordering both docstrings claim.

    Two things make this runnable without a GPU. The training call is replaced by a recorder, so
    "reached training" is a list length rather than 33 GPU-hours. And the reference record is the
    fixture at the arms' own objective, which the harness now shares, so a launch here runs the exact
    configuration a real one would and reaches the identity gate on it rather than on a doctored
    record. The record's identity comes from the registry rather than being restated, so registering
    a superseding run cannot leave these tests green against a value nobody launches with.
    """

    def control_dir(self, tmp_path: Path) -> Path:
        """Where a launch would write its own record, which is also how a refusal is shown to precede it."""
        return tmp_path / "control"

    @pytest.fixture
    def launched(self, monkeypatch: pytest.MonkeyPatch) -> list[ArithmeticTrainConfig]:
        """Stand in for the training call: what it was handed, and whether it was reached at all."""
        configs: list[ArithmeticTrainConfig] = []

        def record_instead_of_training(config: ArithmeticTrainConfig) -> None:
            configs.append(config)

        monkeypatch.setattr(train_neutral, "train_grpo_integer_math", record_instead_of_training)
        return configs

    def launch(self, tmp_path: Path, **overrides: Any) -> None:
        """Run the launcher against a record of the registered reference, overridden per test."""
        registered = NEUTRAL_TASK_ARMS["neutral-arithmetic"].reference_run
        payload = run_config(
            **(
                {
                    "arm": registered.arm,
                    "git_sha": registered.git_sha,
                    "started_at": registered.started_at,
                }
                | overrides
            )
        )
        reference_dir = tmp_path / "reference"
        reference_dir.mkdir()
        train_neutral.main(
            [
                "--arm",
                "neutral-arithmetic",
                "--reference-run-config",
                str(written(reference_dir, payload)),
                "--output-dir",
                str(self.control_dir(tmp_path)),
            ]
        )

    def test_a_launch_past_both_gates_trains_the_matched_shape_and_records_the_pairing(
        self, tmp_path: Path, launched: list[ArithmeticTrainConfig]
    ) -> None:
        """The positive case, which is what makes the refusals' 'nothing ran' assertions mean anything."""
        self.launch(tmp_path)
        assert len(launched) == 1
        assert launched[0].max_steps_full == STEPS
        assert launched[0].num_generations == GROUP
        # What reaches training is the arms' objective, not merely something the check let past.
        assert launched[0].beta == REFERENCE_BETA
        assert launched[0].learning_rate == REFERENCE_LEARNING_RATE
        record = json.loads(
            (self.control_dir(tmp_path) / RUN_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        assert record["compute_match"]["run_identity"] == dataclasses.asdict(
            NEUTRAL_TASK_ARMS["neutral-arithmetic"].reference_run
        )

    def test_another_run_of_the_reference_arm_is_refused_and_nothing_is_trained(
        self, tmp_path: Path, launched: list[ArithmeticTrainConfig]
    ) -> None:
        """Every axis matches here, so this is the launcher's only guard against the wrong run."""
        with pytest.raises(ValueError, match="registered against"):
            self.launch(tmp_path, git_sha="0000000")
        assert launched == []
        # The launcher mkdirs and writes its record only after both gates, so an absent record is how
        # a refusal is shown to have preceded the run rather than followed it.
        assert not (self.control_dir(tmp_path) / RUN_CONFIG_FILENAME).exists()

    def test_an_axis_mismatch_is_reported_instead_of_the_identity_one(
        self, tmp_path: Path, launched: list[ArithmeticTrainConfig]
    ) -> None:
        """The documented order: the actionable axis list first, identity for the record that passes it.

        The axis mismatch is manufactured on the record's side, a reference that trained hotter than
        the harness does, since the harness itself now matches the arms and offers no mismatch to use.
        """
        with pytest.raises(ValueError, match="not compute-matched") as raised:
            self.launch(
                tmp_path,
                git_sha="0000000",
                **{"config.learning_rate": A_HOTTER_LEARNING_RATE},
            )
        assert "learning rate" in reported_problems(raised.value)
        assert "registered against" not in str(raised.value)
        assert launched == []
        assert not (self.control_dir(tmp_path) / RUN_CONFIG_FILENAME).exists()

    def test_a_record_of_a_different_arm_is_refused_by_the_launcher(
        self, tmp_path: Path, launched: list[ArithmeticTrainConfig]
    ) -> None:
        """Refused where the record is read, before either gate, since its numbers cannot tell."""
        with pytest.raises(ValueError, match=re.escape("is a run of 'twin-pd-self'")):
            self.launch(tmp_path, arm="twin-pd-self")
        assert launched == []
        assert not (self.control_dir(tmp_path) / RUN_CONFIG_FILENAME).exists()
