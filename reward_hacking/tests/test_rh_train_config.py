"""The launch-time refusals, and the parity that keeps this trainer honest against the games one.

Offline and CPU-only: nothing here loads weights. Two groups of tests.

The refusals are each about a default that produces a run which looks fine and is not -- a checkpoint
cadence that saves nothing before a spot reclaim, a rotation limit that deletes what the ladder needs,
two arms overwriting each other in one S3 prefix, a completion budget below the coding measurement,
and the slow generation path taken by accident.

The parity test guards the ONE helper this module still duplicates deliberately. ``_build_grpo_config``
here is a sibling of ``games.train._build_grpo_config`` rather than a call into it, because that one is
typed over the games arm registry -- so the science-bearing TRL fields the two must agree on are
asserted equal, and a drift between them fails a test instead of quietly making two projects'
estimators different.

That count used to read "this module's one real duplication" and was false at five: ``resolve_tokenizer``
(now in ``games.preflight``), ``build_callbacks``, ``write_json`` and ``check_built_trainer`` (now in
``games.train``) were copy-pasted with no rationale and no drift guard, and the ``write_json`` pair had
already drifted -- the games copy wrote without an explicit encoding. They are one function each now,
which is why only ``_build_grpo_config`` needs a parity test: a comparison test is what you write when
you cannot share, and four of the five could be shared.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
import torch
from rich.console import Console
from trl.trainer.utils import print_prompt_completions_sample

from games import preflight, provenance, sizing
from games import train as game_train
from grpo.estimator_defaults import GRPO_LOSS_TYPE, GRPO_SCALE_REWARDS
from reward_hacking import train as rh_train
from reward_hacking import train_sequence
from reward_hacking.train_dataset import (
    ARM_CONTROL,
    ARM_LEGIBLE_SUBSET,
    ARM_MISSPECIFIED,
    SOLUTION_PARSER,
)
from reward_hacking.train_partition import (
    SPLIT_ONEOFF,
    SPLIT_ORIGINAL,
    SPLIT_SUBSET3_STRATIFIED,
)
from reward_hacking.train_termination import required_coding_completion_budget

FIXED_TIMESTAMP = datetime(2026, 8, 24, 12, 30, 0, tzinfo=UTC)
CODING_BUDGET = required_coding_completion_budget(rh_train.DEFAULT_MODEL_ID)

# The TRL fields both trainers must set identically, because each is a statement about the estimator
# or about what a run retains, and a silent divergence would make two projects' numbers incomparable.
SHARED_TRL_FIELDS = (
    "loss_type",
    "scale_rewards",
    "epsilon",
    "beta",
    "temperature",
    "top_p",
    "top_k",
    "mask_truncated_completions",
    "log_completions",
    "logging_first_step",
    "eval_strategy",
    "save_strategy",
    "logging_strategy",
    "use_liger_kernel",
    "gradient_checkpointing",
    "report_to",
    "lr_scheduler_type",
    "learning_rate",
    "warmup_steps",
)


def make_config(**overrides: object) -> rh_train.RewardHackingTrainConfig:
    """Build a valid config for tests, overriding whatever the test cares about."""
    base: dict[str, object] = {
        "arm": ARM_MISSPECIFIED,
        "vllm_colocate": False,
        "allow_hf_generation": True,
    }
    settings = base | overrides
    if "s3_dest" not in settings and not settings.get("smoke"):
        # A real arm is refused without somewhere to ship its checkpoints, and the destination has to
        # name THIS arm, so it is derived here rather than written down -- otherwise every control-arm
        # test would fail the per-arm prefix check instead of testing what it meant to.
        arm = f"{settings['arm']}-{rh_train.path_safe_model_id(rh_train.DEFAULT_MODEL_ID)}"
        settings["s3_dest"] = f"s3://bucket/option3/{arm}"
    return rh_train.RewardHackingTrainConfig(**settings)  # pyright: ignore[reportArgumentType]


def hybrid_text_config(n_blocks: int = 8) -> SimpleNamespace:
    """A Qwen3.5-shaped config: three linear-attention layers per full-attention layer."""
    return SimpleNamespace(
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        * n_blocks,
        num_key_value_heads=4,
        num_attention_heads=16,
        hidden_size=2048,
        head_dim=128,
        num_hidden_layers=4 * n_blocks,
        vocab_size=248_320,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
    )


def make_plan(**overrides: object) -> sizing.SizingPlan:
    """Build a sizing plan without touching a GPU or a checkpoint."""
    kwargs: dict[str, object] = {
        "num_generations": 8,
        "prompts_per_step": 8,
        "micro_batch_size": None,
        "max_prompt_tokens": 1024,
        "max_completion_tokens": 1024,
        "cost": sizing.sequence_cost(hybrid_text_config()),
        "free_vram_gib": 44.0,
        "weights_gib": 8.0,
    }
    return sizing.plan_sizing(**(kwargs | overrides))  # pyright: ignore[reportArgumentType]


class TestArmRegistry:
    def test_each_arm_trains_the_split_it_names(self):
        assert make_config(arm=ARM_MISSPECIFIED).split == SPLIT_ONEOFF
        assert make_config(arm=ARM_CONTROL).split == SPLIT_ORIGINAL

    def test_the_legible_subset_arm_is_accepted_and_trains_its_own_split(self):
        """The legibility design's reward side: visible k=3 subset rewarded, hidden check never.

        Refused here until 2026-08-27, deliberately, while training was ungated; the owner approved
        training conditional on the estimator pin, so the trainer now accepts every arm the corpus
        machinery can build (``SPLIT_BY_TRAINABLE_ARM``), not only the frozen flagship pair.
        """
        config = make_config(arm=ARM_LEGIBLE_SUBSET)
        assert config.split == SPLIT_SUBSET3_STRATIFIED
        assert config.arm_tag == f"{ARM_LEGIBLE_SUBSET}-Qwen-Qwen3.5-4B"

    def test_an_unknown_arm_is_refused(self):
        with pytest.raises(ValueError, match="unknown arm"):
            make_config(arm="conflicting")

    def test_a_split_name_is_not_an_arm(self):
        """The likeliest slip once three arms exist: naming the split where the arm belongs."""
        with pytest.raises(ValueError, match="unknown arm"):
            make_config(arm=SPLIT_SUBSET3_STRATIFIED)

    def test_the_arm_tag_names_both_the_arm_and_the_model(self):
        assert make_config().arm_tag == f"{ARM_MISSPECIFIED}-Qwen-Qwen3.5-4B"


class TestEstimatorPin:
    """No arm trains on an unpinned or misdescribed estimator -- the owner's condition on training.

    The values come from ``grpo.estimator_defaults``, the one place the repo decides them, and they
    are refused at config construction rather than left to TRL, which raises on an unknown
    ``loss_type`` only at the first loss computation -- after the model load, on a rented card.
    """

    def test_the_default_estimator_is_the_repo_pin(self):
        config = make_config()
        assert config.loss_type == GRPO_LOSS_TYPE == "dr_grpo"
        assert config.scale_rewards == GRPO_SCALE_REWARDS == "none"

    def test_an_unknown_loss_type_is_refused_at_config_time(self):
        """Sabotage: the transposition typo TRL would only surface after the weights loaded."""
        with pytest.raises(ValueError, match="dpao"):
            make_config(loss_type="dpao")

    def test_an_unknown_scale_rewards_mode_is_refused_at_config_time(self):
        with pytest.raises(ValueError, match="scale_rewards"):
            make_config(scale_rewards="batch-normalised")

    def test_a_liger_unfaithful_loss_type_is_refused_without_acknowledgement(self):
        """`dapo` under Liger executes per-sequence GRPO, not what its name says."""
        with pytest.raises(ValueError, match="num_items_in_batch"):
            make_config(loss_type="dapo")

    def test_the_acknowledgement_unblocks_the_unfaithful_combination(self):
        config = make_config(loss_type="dapo", acknowledge_liger_estimator_mismatch=True)
        assert config.loss_type == "dapo"

    @pytest.mark.parametrize("acknowledged", [False, True])
    def test_luspo_under_liger_is_refused_whether_or_not_acknowledged(self, acknowledged: bool):
        """The padding trim makes Liger's luspo follow each micro-batch's trimmed width.

        Liger's luspo divides by the completion width it is handed without applying the loss mask,
        so it is the one estimator the trim changes; the acknowledgement that unblocks `dapo` above
        would record an executed estimator ("unmasked pad-inclusive aggregation") that no longer
        describes what runs. Refused ahead of the acknowledgement check, so the message says why the
        flag would not help instead of inviting it.
        """
        with pytest.raises(ValueError, match="trims every training micro-batch"):
            make_config(loss_type="luspo", acknowledge_liger_estimator_mismatch=acknowledged)

    def test_luspo_without_liger_still_constructs(self):
        """TRL's own luspo masks before it aggregates, so the trim leaves it alone."""
        config = make_config(loss_type="luspo", use_liger_kernel=False)
        assert (config.loss_type, config.use_liger_kernel) == ("luspo", False)

    def test_the_estimator_is_pinned_across_a_resume(self):
        """A resumed run on a different estimator is two experiments under one set of steps."""
        assert "loss_type" in rh_train.RESUME_IDENTITY_FIELDS
        assert "scale_rewards" in rh_train.RESUME_IDENTITY_FIELDS


class TestCheckpointRetention:
    def test_per_step_saving_and_no_rotation_are_the_defaults(self):
        config = make_config()
        assert config.save_steps == 1
        assert config.save_total_limit == 0

    def test_a_coarser_cadence_is_refused_for_a_real_arm(self):
        """Sabotage: the games default of 5, which lost three reclaimed runs entirely."""
        with pytest.raises(ValueError, match="save_steps=5"):
            make_config(save_steps=5)

    def test_the_games_default_cadence_is_also_refused(self):
        with pytest.raises(ValueError, match="save_steps=10"):
            make_config(save_steps=10)

    def test_a_rotation_limit_that_would_delete_checkpoints_is_refused(self):
        """Sabotage: the games default of 50, which deletes the 20 oldest of 70."""
        with pytest.raises(ValueError, match="oldest of 70 checkpoints"):
            make_config(save_total_limit=50)

    def test_a_rotation_limit_at_or_above_the_checkpoint_count_is_allowed(self):
        assert make_config(save_total_limit=70).save_total_limit == 70

    def test_a_smoke_run_may_save_less_often_because_it_is_not_a_result(self):
        assert make_config(smoke=True, save_steps=2).save_steps == 2

    def test_a_zero_cadence_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            make_config(save_steps=0)


class TestCompletionBudget:
    def test_the_default_comes_from_the_coding_screen_not_the_game_screen(self):
        assert make_config().completion_budget == CODING_BUDGET
        assert CODING_BUDGET > game_train.MEASURED_TERMINATION_BUDGET

    def test_a_budget_below_the_coding_measurement_is_refused(self):
        """Sabotage: the 8,192 the plan originally priced the arms at."""
        with pytest.raises(ValueError, match="needs at least"):
            make_config(max_completion_tokens=8192)

    def test_the_game_prompt_floor_is_also_refused_here(self):
        with pytest.raises(ValueError, match="needs at least"):
            make_config(max_completion_tokens=game_train.MEASURED_TERMINATION_BUDGET)

    def test_a_timing_probe_may_say_so_deliberately(self):
        config = make_config(max_completion_tokens=2048, allow_short_completions=True)
        assert config.completion_budget == 2048

    def test_a_thinking_off_plumbing_run_is_exempt(self):
        assert make_config(max_completion_tokens=2048, thinking=False).completion_budget == 2048


class TestGenerationPath:
    def test_the_slow_path_is_refused_unless_stated(self):
        """Sabotage: forget --allow-hf-generation, and the 5.5x is refused rather than paid."""
        with pytest.raises(ValueError, match="72 minutes per step"):
            rh_train.RewardHackingTrainConfig(
                arm=ARM_MISSPECIFIED, vllm_colocate=False, allow_hf_generation=False
            )

    def test_an_out_of_range_engine_share_is_refused(self):
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            make_config(vllm_colocate=True, vllm_gpu_memory_utilization=1.0)

    def test_the_importance_sampling_correction_is_refused_at_this_budget(self):
        """The refusal that saves a mid-run out-of-memory: 24,576 tokens is 22.7 GiB of fp32 logits."""
        with pytest.raises(ValueError, match="importance-sampling correction ON"):
            make_config(
                vllm_colocate=True,
                allow_hf_generation=False,
                vllm_importance_sampling_correction=True,
            )

    def test_dropping_the_correction_makes_the_budget_allowed(self):
        config = make_config(
            vllm_colocate=True,
            allow_hf_generation=False,
            vllm_importance_sampling_correction=False,
        )
        assert config.vllm_colocate is True
        assert config.vllm_max_model_length == config.max_prompt_tokens + CODING_BUDGET


class TestTheColocateBannerRunsAgainstThisThreadsConfig:
    """`_derive_plan` announces the estimator through `games.train.log_colocate_settings`.

    The banner is the games trainer's, and it is called with THIS thread's config, which carries
    neither of the two instrument knobs games added for its own probe (`old_logps_chunk_tokens`,
    `vllm_importance_sampling_log_only`). While it read them off the config behind a
    `cast("Any", config)`, every launch here with the correction ON -- its default, and exempt from the
    memory refusal on the smoke path -- died with an `AttributeError` at the banner before any weights
    loaded. Nothing covered it: the only test that reached `_prepare_run` tripped earlier, at
    `read_recorded_launch`. So the banner takes those values as arguments now, and this drives the real
    call rather than the banner directly, because a hand-written argument list would go green while the
    call site drifted.

    Sabotage run once and watched red (2026-09-04): making the banner read `old_logps_chunk_tokens`
    off the config again, behind the cast it used to carry, failed two tests here with the original
    `AttributeError`.
    """

    def derive(self, monkeypatch: pytest.MonkeyPatch, **overrides: object) -> sizing.SizingPlan:
        """Run `_derive_plan` with the checkpoint read stubbed; everything else is production code."""
        monkeypatch.setattr(
            rh_train,
            "checkpoint_sequence_cost",
            lambda _model_id: sizing.sequence_cost(hybrid_text_config()),
        )
        return rh_train._derive_plan(  # pyright: ignore[reportPrivateUsage]
            make_config(smoke=True, **overrides),
            device={
                "total_vram_gib": 48.0,
                "free_vram_gib_at_start": 44.0,
                "device_name": "NVIDIA L40S",
            },
            dtype=torch.bfloat16,
            param_count=4_000_000_000,
        )

    def test_the_correction_on_by_default_reaches_a_plan_rather_than_an_attribute_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        assert (
            next(
                f
                for f in fields(rh_train.RewardHackingTrainConfig)
                if f.name == "vllm_importance_sampling_correction"
            ).default
            is True
        )
        with caplog.at_level(logging.INFO, logger=game_train.logger.name):
            plan = self.derive(monkeypatch)
        assert plan.micro_batch_size >= 1
        assert "correction is ON" in caplog.text

    def test_the_banner_describes_the_trainer_class_this_thread_builds(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """`_build_trainer` passes neither instrument keyword, so the class defaults are what runs."""
        with caplog.at_level(logging.INFO, logger=game_train.logger.name):
            self.derive(monkeypatch)
        assert str(game_train.PaddingTrimmedGRPOTrainer.old_logps_chunk_tokens) in caplog.text
        assert game_train.PaddingTrimmedGRPOTrainer.importance_sampling_log_only is False
        assert "log-only importance sampling" not in caplog.text

    def test_the_correction_off_branch_states_its_caveat(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.INFO, logger=game_train.logger.name):
            self.derive(monkeypatch, vllm_importance_sampling_correction=False)
        assert "correction is OFF" in caplog.text

    def test_this_config_carries_none_of_the_games_instrument_fields(self):
        """The fact the contract exists for, asserted rather than assumed: adding one here would make
        the regression untestable by making the two configs accidentally interchangeable again."""
        names = {f.name for f in fields(rh_train.RewardHackingTrainConfig)}
        assert not names & {
            "old_logps_chunk_tokens",
            "vllm_importance_sampling_log_only",
            "cast_lm_head_to_fp32",
        }


class TestS3Destination:
    def test_a_destination_naming_this_arm_is_accepted(self):
        config = make_config(s3_dest="s3://bucket/prefix/misspecified-Qwen-Qwen3.5-4B")
        assert config.s3_dest.endswith(config.arm_tag)

    def test_a_shared_prefix_is_refused_and_the_message_names_the_fix(self):
        """Sabotage: point both arms at one prefix, where the second silently replaces the first."""
        with pytest.raises(ValueError, match="does not end in this arm's own segment") as caught:
            make_config(s3_dest="s3://bucket/option3")
        assert "s3://bucket/option3/misspecified-Qwen-Qwen3.5-4B" in str(caught.value)

    def test_the_other_arms_prefix_is_refused_for_this_arm(self):
        with pytest.raises(ValueError, match="does not end in this arm's own segment"):
            make_config(arm=ARM_CONTROL, s3_dest="s3://bucket/option3/misspecified-Qwen-Qwen3.5-4B")

    def test_a_non_s3_uri_is_refused_at_startup(self):
        with pytest.raises(ValueError, match="must be an s3:// URI"):
            make_config(s3_dest="/local/path")

    def test_a_real_arm_with_no_destination_is_refused(self):
        """Replaces an earlier assertion that no destination was fine. Retention is a requirement:
        the ladder is read at every rung and a reclaim takes the instance's disk with it."""
        with pytest.raises(ValueError, match="no --s3-dest"):
            rh_train.RewardHackingTrainConfig(
                arm=ARM_MISSPECIFIED, vllm_colocate=False, allow_hf_generation=True
            )


class TestResumeAddressability:
    def test_resuming_the_latest_checkpoint_needs_an_explicit_output_dir(self):
        with pytest.raises(ValueError, match="needs an explicit --output-dir"):
            make_config(resume_from_checkpoint=rh_train.RESUME_LATEST)

    def test_naming_the_output_dir_makes_it_addressable(self):
        config = make_config(
            resume_from_checkpoint=rh_train.RESUME_LATEST, output_dir="artifacts/x"
        )
        assert config.resume_from_checkpoint == rh_train.RESUME_LATEST


class TestTheLaunchResumesThroughTheCompletenessGate:
    """`_prepare_run` resolves its resume through `games.train.resolve_complete_resume_checkpoint`.

    The resolver is shared with the games trainer and tested there; what this trainer owns is the
    call. A revert to the plain resolver would leave every resolver test green while a torn
    checkpoint resumed with a freshly constructed optimizer, so the launch itself runs here. The jail
    probe is the one stub before the resolver. `read_recorded_launch`, the first thing the launch
    does with the checkpoint it resolved, is a tripwire that records which checkpoint arrived and
    stops the launch before it reads the card or loads a tokenizer.
    """

    def seed_a_checkpoint(self, run_dir: Path, step: int, *, omit: tuple[str, ...] = ()) -> Path:
        """Every file the completeness gate requires, minus `omit`; the reclaim-mid-sync shape."""
        checkpoint = run_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / game_train.TRAINER_STATE_FILENAME).write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
        for name in (
            *game_train.REQUIRED_CHECKPOINT_FILES,
            game_train.INIT_ADAPTER_WEIGHTS_FILENAME,
        ):
            if name not in omit and name != game_train.TRAINER_STATE_FILENAME:
                (checkpoint / name).write_bytes(b"")
        return checkpoint

    def launch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        """Run `_prepare_run` against `tmp_path / "run"`; return the checkpoint it went on to read."""
        read_for: list[str] = []

        def trip(output_dir: str, *, checkpoint: str) -> dict[str, object]:
            del output_dir
            read_for.append(checkpoint)
            raise RuntimeError("tripwire: the launch reached its recorded-launch read")

        monkeypatch.setattr(rh_train, "assert_jail_usable", lambda **_kwargs: {"stubbed": True})
        monkeypatch.setattr(rh_train, "read_recorded_launch", trip)
        config = make_config(
            smoke=True,
            resume_from_checkpoint=rh_train.RESUME_LATEST,
            output_dir=str(tmp_path / "run"),
            grader_scratch_root=str(tmp_path / "grader-scratch"),
        )
        with pytest.raises(RuntimeError, match="tripwire"):
            rh_train._prepare_run(config, kernel_bridge=None)
        (checkpoint,) = read_for
        return checkpoint

    def test_a_complete_newest_checkpoint_is_the_one_the_launch_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The positive control on the gate: nothing complete is stepped past or moved."""
        run_dir = tmp_path / "run"
        self.seed_a_checkpoint(run_dir, 2)
        newest = self.seed_a_checkpoint(run_dir, 3)
        assert self.launch(tmp_path, monkeypatch) == str(newest)
        assert newest.is_dir()
        assert not (run_dir / game_train.INCOMPLETE_CHECKPOINTS_DIRNAME).exists()

    def test_a_torn_newest_checkpoint_is_set_aside_before_the_launch_reads_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """`checkpoint-3` without its optimizer: the launch reads `checkpoint-2`'s record instead,
        the torn directory is out of the `checkpoint-*` glob, and the log names the fallback."""
        run_dir = tmp_path / "run"
        complete = self.seed_a_checkpoint(run_dir, 2)
        torn = self.seed_a_checkpoint(run_dir, 3, omit=("optimizer.pt",))
        with caplog.at_level(logging.WARNING, logger=game_train.logger.name):
            resumed = self.launch(tmp_path, monkeypatch)
        assert resumed == str(complete)
        assert not torn.exists()
        (moved,) = (run_dir / game_train.INCOMPLETE_CHECKPOINTS_DIRNAME).glob("*/checkpoint-3")
        assert (moved / game_train.TRAINER_STATE_FILENAME).is_file()
        assert "INCOMPLETE CHECKPOINT: checkpoint-3 lacks ['optimizer.pt']" in caplog.text


class TestCompletionLogging:
    """The rollout trace is kept; the per-step rich rendering of every completion is not.

    TRL's default `num_completions_to_print=None` renders all completions of a step as a table on
    stdout: roughly 300,000 lines and 27 MB per step at 64 completions, 8-12 s of the post-step
    phase, and a 70-step log in which ~650 timestamped lines sit under 17.8 million others. The
    parquet under `<output_dir>/completions/` is written in the same block regardless of the count.
    """

    def test_the_trace_is_kept_and_nothing_is_rendered(self):
        config = rh_train._build_grpo_config(
            make_config(output_dir="artifacts/rh-logging"), make_plan(), dtype=torch.float32
        )
        assert config.log_completions is True
        assert config.num_completions_to_print == 0

    def test_trl_treats_zero_as_render_nothing_not_render_all(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Pins the TRL semantics the saving rests on: None means all, 0 means none. A TRL upgrade
        that made 0 mean "all" would silently bring the 8-12 s and the 27 MB per step back."""
        printed: list[object] = []
        monkeypatch.setattr(Console, "print", lambda _self, *args, **_kwargs: printed.extend(args))

        def render(num_samples: int | None) -> None:
            """TRL annotates `num_samples` as `int` while defaulting it to None, hence the ignore."""
            print_prompt_completions_sample(
                ["prompt"],
                ["completion"],
                {"reward": [1.0]},
                [0.0],
                7,
                num_samples,  # pyright: ignore[reportArgumentType]
            )

        render(0)
        assert printed == []
        render(None)
        assert len(printed) == 1


class TestGrpoConfigParity:
    """The science-bearing TRL fields must match the games trainer exactly, field by field."""

    def _both(self) -> tuple[Any, Any]:
        plan = make_plan()
        mine = rh_train._build_grpo_config(
            make_config(output_dir="artifacts/rh-parity"), plan, dtype=torch.float32
        )
        theirs = game_train._build_grpo_config(
            game_train.GameTrainConfig(
                arm="twin-pd-group",
                generate_fresh=True,
                output_dir="artifacts/games-parity",
                learning_rate=make_config().learning_rate,
                lr_scheduler=make_config().lr_scheduler,
                warmup_ratio=make_config().warmup_ratio,
            ),
            plan,
            dtype=torch.float32,
        )
        return mine, theirs

    @pytest.mark.parametrize("field", SHARED_TRL_FIELDS)
    def test_the_shared_field_matches_the_games_trainer(self, field: str):
        mine, theirs = self._both()
        assert getattr(mine, field) == getattr(theirs, field), field

    def test_the_retention_fields_deliberately_do_not_match(self):
        """The one intended divergence, asserted so it cannot be undone by accident."""
        mine, theirs = self._both()
        assert (mine.save_steps, mine.save_total_limit) == (1, 0)
        assert (theirs.save_steps, theirs.save_total_limit) == (10, 50)

    def test_the_run_name_is_the_arm_tag_so_two_arms_are_never_confused(self):
        mine, _ = self._both()
        assert mine.run_name == make_config().arm_tag

    def test_the_completion_length_is_the_coding_budget(self):
        mine, _ = self._both()
        assert mine.max_completion_length == CODING_BUDGET

    def test_the_model_dtype_goes_through_trls_own_key(self):
        """`torch_dtype` is ignored by TRL, so a model would load in float32 under a bf16 trainer."""
        mine, _ = self._both()
        init_kwargs = cast("dict[str, object]", mine.model_init_kwargs)
        assert init_kwargs["dtype"] == torch.float32
        assert "torch_dtype" not in init_kwargs


class TestSmokeShrinkAndOutputDir:
    def test_the_smoke_keeps_checkpointing_so_it_proves_the_retention_seam(self):
        shrunk = rh_train.shrink_for_smoke(make_config())
        assert shrunk.save_steps <= shrunk.max_steps
        assert shrunk.smoke is True

    def test_the_output_directory_names_the_arm_the_model_and_the_launch(self):
        directory = rh_train.default_output_dir(make_config(), timestamp=FIXED_TIMESTAMP)
        assert directory.endswith("misspecified-Qwen-Qwen3.5-4B-20260824-123000")

    def test_a_smoke_run_directory_is_labelled_as_one(self):
        directory = rh_train.default_output_dir(
            make_config(), timestamp=FIXED_TIMESTAMP, smoke=True
        )
        assert "/smoke-" in directory


class TestCliDefaults:
    def test_the_cli_defaults_to_per_step_retention_without_rotation(self):
        config = rh_train._parse_args(
            [
                "--arm",
                ARM_MISSPECIFIED,
                "--no-vllm-colocate",
                "--allow-hf-generation",
                "--s3-dest",
                "s3://bucket/option3/misspecified-Qwen-Qwen3.5-4B",
            ]
        )
        assert (config.save_steps, config.save_total_limit) == (1, 0)
        assert config.seed == 0
        assert config.max_steps == 70
        assert config.num_generations == 8
        assert config.prompts_per_step == 8

    def test_the_cli_refuses_a_shared_s3_prefix_before_anything_loads(self):
        with pytest.raises(ValueError, match="does not end in this arm's own segment"):
            rh_train._parse_args(
                [
                    "--arm",
                    ARM_MISSPECIFIED,
                    "--no-vllm-colocate",
                    "--allow-hf-generation",
                    "--s3-dest",
                    "s3://bucket/option3",
                ]
            )

    def test_the_cli_accepts_the_legible_subset_arm(self):
        """`--arm` gates through argparse choices, a separate gate from the config's own refusal."""
        config = rh_train._parse_args(
            [
                "--arm",
                ARM_LEGIBLE_SUBSET,
                "--no-vllm-colocate",
                "--allow-hf-generation",
                "--s3-dest",
                f"s3://bucket/option3/{ARM_LEGIBLE_SUBSET}-Qwen-Qwen3.5-4B",
            ]
        )
        assert config.arm == ARM_LEGIBLE_SUBSET
        assert config.split == SPLIT_SUBSET3_STRATIFIED
        assert (config.max_steps, config.save_steps, config.save_total_limit) == (70, 1, 0)

    def test_the_cli_still_refuses_an_untrainable_arm_name(self):
        """argparse choices exit with code 2 on a value outside the trainable registry."""
        with pytest.raises(SystemExit):
            rh_train._parse_args(["--arm", "conflicting"])


class TestTheLaunchGateRefusesRatherThanWarns:
    """An arm may not be built without a gradient screen for it on disk."""

    def _plan_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stages: str) -> None:
        monkeypatch.setenv("RH_OPTION3_ARM", ARM_MISSPECIFIED)
        monkeypatch.setenv("RH_OPTION3_STAGES", stages)
        monkeypatch.setattr(train_sequence, "SCREEN_DIR", tmp_path / "screen")

    def _write_verdict(self, *, solution_parser: str | None) -> None:
        """Lay down a screen verdict; ``None`` omits the key, as every pre-fix screen does."""
        artifact = train_sequence.screen_artifact()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        record = {} if solution_parser is None else {"solution_parser": solution_parser}
        artifact.write_text(json.dumps(record), encoding="utf-8")

    def test_an_arm_with_no_screen_on_disk_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Sabotage: launch the arm blind, exactly as an unwatched rented box would."""
        self._plan_env(monkeypatch, tmp_path, "arm")
        with pytest.raises(ValueError, match="no gradient screen"):
            train_sequence.stages()

    def test_selecting_the_screen_in_the_same_invocation_satisfies_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        self._plan_env(monkeypatch, tmp_path, "screen,arm")
        assert [stage.name.split("-")[0] for stage in train_sequence.stages()] == ["screen", "arm"]

    def test_an_existing_screen_artifact_satisfies_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        self._plan_env(monkeypatch, tmp_path, "arm")
        self._write_verdict(solution_parser=SOLUTION_PARSER)
        assert len(train_sequence.stages()) == 1

    def test_a_screen_artifact_from_another_parser_does_not_satisfy_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Existence is not enough: a verdict measured under a retired parser describes another arm.

        A missing key is what every verdict written before the parser change carries, which is the
        shape of the two sitting in the live run's S3 prefix. Covered in depth in
        ``test_rh_train_sequence``; asserted here too because this class is the one that reads as
        "what satisfies the launch gate".
        """
        self._plan_env(monkeypatch, tmp_path, "arm")
        self._write_verdict(solution_parser=None)
        with pytest.raises(ValueError, match="RE-SCREEN"):
            train_sequence.stages()

    def test_a_screen_only_invocation_needs_no_prior_screen(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        self._plan_env(monkeypatch, tmp_path, "screen")
        assert len(train_sequence.stages()) == 1


class TestRetentionDestination:
    """A real arm may not run with nowhere to ship its checkpoints."""

    def test_an_arm_without_a_destination_is_refused(self):
        """Sabotage: the plan's S3 base unset, which leaves no retention AND no recovery."""
        with pytest.raises(ValueError, match="no --s3-dest"):
            rh_train.RewardHackingTrainConfig(
                arm=ARM_MISSPECIFIED, vllm_colocate=False, allow_hf_generation=True
            )

    def test_a_smoke_run_is_exempt_because_it_is_plumbing(self):
        smoke = rh_train.RewardHackingTrainConfig(
            arm=ARM_MISSPECIFIED, vllm_colocate=False, smoke=True
        )
        assert smoke.s3_dest == ""

    def test_a_destination_naming_the_arm_satisfies_it(self):
        config = make_config(s3_dest="s3://bucket/option3/misspecified-Qwen-Qwen3.5-4B")
        assert config.s3_dest.endswith(config.arm_tag)


class TestResumeIdentityCoversWhatChangesTheExperiment:
    """Each pinned field silently changes what a checkpoint's existing steps mean."""

    @pytest.mark.parametrize(
        "field",
        [
            "arm",
            "model_id",
            "partition_path",
            "max_prompts",
            "max_steps",
            "lora_alpha",
            "lora_dropout",
            "grader_timeout_seconds",
            "seed",
            "max_completion_tokens",
        ],
    )
    def test_the_field_is_pinned_across_a_resume(self, field: str):
        assert field in rh_train.RESUME_IDENTITY_FIELDS

    def test_max_prompts_is_pinned_because_it_changes_which_problems_were_seen(self):
        """The one that would otherwise slip past `partition_path`: same partition, fewer problems."""
        assert "max_prompts" in rh_train.RESUME_IDENTITY_FIELDS

    @pytest.mark.parametrize(
        "field",
        ["temperature", "top_p", "top_k", "learning_rate", "lr_scheduler", "warmup_ratio"],
    )
    def test_the_sampler_and_the_schedule_are_pinned(self, field: str):
        """The set was self-inconsistent: `max_steps` was pinned precisely because the scheduler is
        rebuilt from current arguments on resume, and the scheduler's other three inputs were not."""
        assert field in rh_train.RESUME_IDENTITY_FIELDS

    @pytest.mark.parametrize("field", ["beta", "epsilon", "vllm_importance_sampling_correction"])
    def test_the_objective_itself_is_pinned(self, field: str):
        """Each changes what the surrogate loss IS between one step and the next."""
        assert field in rh_train.RESUME_IDENTITY_FIELDS

    def test_the_adapter_shape_is_pinned_not_only_its_scaling(self):
        """`lora_rank` surfaces as a torch shape error after the model load, not as a refusal."""
        assert "lora_rank" in rh_train.RESUME_IDENTITY_FIELDS


class TestEveryConfigFieldIsClassifiedForResume:
    """The guard against the next setting being silently unguarded, which is how eight came to be.

    Classification by omission is what failed: the sampler and the schedule were simply absent, and
    nothing anywhere said whether that was a decision. Every field now lands in exactly one tuple, and
    the deliberately-movable tuple carries a reason per entry, so adding a setting forces the question.
    """

    def field_names(self) -> set[str]:
        return {field.name for field in fields(rh_train.RewardHackingTrainConfig)}

    def classification(self) -> dict[str, tuple[str, ...]]:
        return {
            "identity": rh_train.RESUME_IDENTITY_FIELDS,
            "pinned indirectly": rh_train.RESUME_FIELDS_PINNED_INDIRECTLY,
            "movable": rh_train.RESUME_MOVABLE_FIELDS,
        }

    def test_every_field_is_classified(self):
        classified = {name for tuples in self.classification().values() for name in tuples}
        assert self.field_names() - classified == set()

    def test_no_field_is_classified_twice(self):
        counted: dict[str, list[str]] = {}
        for label, names in self.classification().items():
            for name in names:
                counted.setdefault(name, []).append(label)
        assert {name: labels for name, labels in counted.items() if len(labels) > 1} == {}

    def test_no_tuple_names_a_field_that_does_not_exist(self):
        """A rename would otherwise leave a pin that silently guards nothing."""
        for label, names in self.classification().items():
            assert set(names) <= self.field_names(), (label, set(names) - self.field_names())

    def test_the_sizing_tuple_names_sizing_plan_fields_rather_than_config_fields(self):
        """Two different objects: the config holds the REQUEST, the plan holds what autosize derived."""
        plan_fields = {field.name for field in fields(sizing.SizingPlan)}
        assert set(rh_train.RESUME_SIZING_FIELDS) <= plan_fields

    def test_the_provenance_tuple_names_top_level_record_keys(self):
        """These are siblings of `config` in run_config.json, which is why nothing compared them."""
        assert set(rh_train.RESUME_PROVENANCE_FIELDS) == {"git_sha", "executed_estimator"}
        assert set(rh_train.RESUME_PROVENANCE_FIELDS).isdisjoint(self.field_names())


class TestResumeRefusesCodeItCannotAttribute:
    """`git_sha` was recorded at the top level of the run record and never compared to anything.

    The fake provenance here returns ONLY the two keys the real function returns, and that shape
    IS a regression test: an earlier fixture smuggled an ``executed_estimator`` key into the fake,
    a shape ``games.provenance.git_provenance`` never produces, so every test passed while the
    real gate raised ``KeyError`` on every resume that cleared the sha sentinel.
    """

    def recorded(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "git_sha": "a" * 40,
            "git_tree_dirty": False,
            "executed_estimator": "dr_grpo (faithful under Liger)",
            "sizing_plan": {"micro_batch_size": 1},
        }
        return base | overrides

    def current(self, monkeypatch: pytest.MonkeyPatch, sha: str, *, dirty: bool = False) -> None:
        monkeypatch.setattr(
            rh_train,
            "git_provenance",
            lambda: {"git_sha": sha, "git_tree_dirty": dirty},
        )

    def test_the_fake_returns_exactly_the_real_functions_keys(self):
        """Pin the fake's shape to the real function, so the mock cannot drift apart again."""
        assert set(provenance.git_provenance()) == {"git_sha", "git_tree_dirty"}

    def test_a_matching_sha_is_accepted_and_reported(self, monkeypatch: pytest.MonkeyPatch):
        self.current(monkeypatch, "a" * 40)
        checked = rh_train.assert_resume_provenance_matches(
            self.recorded(), checkpoint="checkpoint-30", config=make_config()
        )
        assert checked["git_sha"] == "a" * 40
        assert checked["compared"] == list(rh_train.RESUME_PROVENANCE_FIELDS)
        # The estimator half is DERIVED from this launch's config plus the recorded micro batch,
        # because git_provenance carries only git keys; the record says what was compared.
        assert checked["executed_estimator"] == "dr_grpo (faithful under Liger)"

    def test_a_differing_sha_refuses_and_names_the_consequence(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Sabotage: relaunch the same command after the freshness floor moved HEAD."""
        self.current(monkeypatch, "b" * 40)
        with pytest.raises(RuntimeError, match="refusing to resume") as caught:
            rh_train.assert_resume_provenance_matches(
                self.recorded(), checkpoint="checkpoint-30", config=make_config()
            )
        assert "different code" in str(caught.value)

    def test_a_changed_executed_estimator_refuses(self, monkeypatch: pytest.MonkeyPatch):
        self.current(monkeypatch, "a" * 40)
        with pytest.raises(RuntimeError, match="refusing to resume"):
            rh_train.assert_resume_provenance_matches(
                self.recorded(executed_estimator="dr_grpo (non-Liger TRL path)"),
                checkpoint="checkpoint-30",
                config=make_config(),
            )

    def test_a_liger_flip_on_the_resume_command_refuses(self, monkeypatch: pytest.MonkeyPatch):
        """The real-world direction: --no-liger on the relaunch changes what would EXECUTE."""
        self.current(monkeypatch, "a" * 40)
        with pytest.raises(RuntimeError, match="executed_estimator"):
            rh_train.assert_resume_provenance_matches(
                self.recorded(),
                checkpoint="checkpoint-30",
                config=make_config(use_liger_kernel=False),
            )

    def test_a_record_with_no_readable_micro_batch_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The executed estimator cannot be derived without the recorded sizing plan."""
        self.current(monkeypatch, "a" * 40)
        for broken in (None, {}, {"micro_batch_size": "one"}):
            with pytest.raises(RuntimeError, match="sizing_plan"):
                rh_train.assert_resume_provenance_matches(
                    self.recorded(sizing_plan=broken),
                    checkpoint="checkpoint-30",
                    config=make_config(),
                )

    @pytest.mark.parametrize("side", ["recorded", "current"])
    def test_an_unattributable_sha_on_either_side_refuses_rather_than_comparing_equal(
        self, side: str, monkeypatch: pytest.MonkeyPatch
    ):
        """The whole reason this is not one plain comparison.

        `git_sha` never raises -- it records the reason in the returned string -- so on a box with no
        git checkout and no GIT_SHA exported, both sides read the same sentinel and a plain comparison
        would pass while answering nothing. 74 artifacts on disk carry exactly that string.
        """
        sentinel = f"{provenance.UNKNOWN_SHA} (git rev-parse failed: not a git repository)"
        self.current(monkeypatch, sentinel if side == "current" else "a" * 40)
        recorded = self.recorded(git_sha=sentinel) if side == "recorded" else self.recorded()
        with pytest.raises(RuntimeError, match="names no commit"):
            rh_train.assert_resume_provenance_matches(
                recorded, checkpoint="checkpoint-30", config=make_config()
            )

    def test_two_sentinels_would_have_compared_equal(self, monkeypatch: pytest.MonkeyPatch):
        """The positive control on the test above: without the sentinel check this passes silently."""
        sentinel = f"{provenance.UNKNOWN_SHA} (git rev-parse failed: not a git repository)"
        self.current(monkeypatch, sentinel)
        recorded = self.recorded(git_sha=sentinel)
        assert recorded["git_sha"] == rh_train.git_provenance()["git_sha"]
        with pytest.raises(RuntimeError, match="names no commit"):
            rh_train.assert_resume_provenance_matches(
                recorded, checkpoint="checkpoint-30", config=make_config()
            )

    def test_a_dirty_tree_warns_rather_than_refusing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """A dirty tree is normal on a shared checkout; a matching sha is just weaker evidence."""
        self.current(monkeypatch, "a" * 40, dirty=True)
        with caplog.at_level(logging.WARNING):
            rh_train.assert_resume_provenance_matches(
                self.recorded(), checkpoint="checkpoint-30", config=make_config()
            )
        assert "DIRTY tree" in caplog.text

    @pytest.mark.parametrize(
        "call",
        [
            "fields=RESUME_IDENTITY_FIELDS",
            "fields=RESUME_SIZING_FIELDS",
            "assert_resume_provenance_matches(",
        ],
    )
    def test_the_resume_path_actually_invokes_every_comparison(self, call: str):
        """A guard nobody calls is the shape of the bug being fixed, not a fix for it.

        `git_sha` was RECORDED all along -- what was missing was the call that compared it. So testing
        the comparison in isolation would reproduce the original defect exactly: a correct function
        nothing reaches. Read off the source because `_prepare_run` proves the jail, loads a tokenizer
        and reads the card, none of which belongs in an offline suite.
        """
        source = Path(str(rh_train.__file__)).read_text(encoding="utf-8")
        body = source.split("def _prepare_run", 1)[1].split("\ndef ", 1)[0]
        assert call in body


class EosStubTokenizer:
    """A checkpoint declaring several stop tokens, which is the shape `repair_scalar_eos` exists for.

    The verified `openbmb/MiniCPM5-1B` case: `generation_config.eos_token_id` is `[1, 130073]` while
    the scalar `eos_token` is id 1, the same token as `pad_token`, and the template terminates every
    turn with 130073. `eos_token` is a property whose setter moves `eos_token_id`, because that is the
    coupling a real tokenizer has and the repair depends on it.
    """

    TOKENS: ClassVar[dict[int, str]] = {1: "</s>", 130073: "<|im_end|>"}

    def __init__(self) -> None:
        self.pad_token_id = 1
        self.pad_token = "</s>"
        self.padding_side = "right"
        self._eos_token_id = 1

    @property
    def eos_token(self) -> str:
        return self.TOKENS[self._eos_token_id]

    @eos_token.setter
    def eos_token(self, token: str) -> None:
        self._eos_token_id = {name: key for key, name in self.TOKENS.items()}[token]

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.TOKENS[token_id]

    def apply_chat_template(self, _conversation: object, **kwargs: object) -> str:
        tail = "<think>\n" if kwargs.get("enable_thinking") else ""
        return f"<|im_start|>assistant\n{tail}"

    def get_chat_template(self) -> str:
        # Must not mention the knob name at all: `resolve_chat_template_kwargs` substring-matches
        # the whole template text, so even prose about it would pin the effort.
        return "{{ messages }}<|im_start|>assistant\n"


class TestBothTrainersShareOneTokenizerResolver:
    """Four helpers were copy-pasted between the two trainers with no rationale and no drift guard.

    Unlike `_build_grpo_config`, which documents its sibling status and is pinned by the parity test
    above, none of the four carried either -- so a fix to the tokenizer and EOS-repair ordering had to
    be made twice or it silently applied to one research thread.

    The functional test of the shared resolver lives HERE, in the reward-hacking suite, on purpose:
    the games suite already had an ordering test for the repair, so a sabotage inside the shared
    function had to be able to redden this suite too, or "they share it" would be an assertion about
    the source rather than about behaviour.
    """

    def test_the_shared_resolver_repairs_an_ambiguous_stop_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        tokenizer = EosStubTokenizer()
        monkeypatch.setattr(
            preflight.AutoTokenizer,
            "from_pretrained",
            classmethod(lambda _cls, *_a, **_k: tokenizer),
        )
        monkeypatch.setattr(
            preflight.GenerationConfig,
            "from_pretrained",
            classmethod(lambda _cls, *_a, **_k: SimpleNamespace(eos_token_id=[1, 130073])),
        )
        resolved, facts = preflight.resolve_tokenizer("stub/multi-eos", thinking=True)
        scalar_eos = cast("dict[str, object]", facts["scalar_eos"])
        assert scalar_eos["repaired"] is True
        # The real terminator, not the pad token: unrepaired, nothing ever halts.
        assert resolved.eos_token_id == 130073  # pyright: ignore[reportAttributeAccessIssue]
        assert resolved.padding_side == "left"
        assert facts["prefilled_think"] is True
        assert facts["chat_template_kwargs"] == {}

    # Every helper the reward-hacking trainer calls out of the games code, each with its canonical
    # home. The resume resolver, the already-complete recogniser, the rollout-trace verifier and the
    # trainer class itself joined the original four as they were shared; the reward-hacking trainer
    # once carried its own resume resolver by the same name, and then its own trainer subclass
    # (`TraceGuardedGRPOTrainer`, a second copy of the trace guard's `log`), which are exactly the
    # copies this list exists to keep out.
    SHARED_HELPERS: ClassVar[tuple[tuple[str, ModuleType], ...]] = (
        ("resolve_tokenizer", preflight),
        ("verify_trace_files", preflight),
        ("build_callbacks", game_train),
        ("write_json", game_train),
        ("check_built_trainer", game_train),
        ("resolve_complete_resume_checkpoint", game_train),
        ("completed_run", game_train),
        ("PaddingTrimmedGRPOTrainer", game_train),
    )
    # Reached only through a helper above, so the reward-hacking trainer binds no name for them and
    # only the no-private-copy check applies: `resolve_complete_resume_checkpoint` calls the first;
    # `PaddingTrimmedGRPOTrainer._prepare_inputs` calls the trim and its `log` calls the trace guard
    # with the trace path, which is why the reward-hacking trainer may not grow a `log` or a
    # `_prepare_inputs` of its own either (`test_the_reward_hacking_trainer_defines_no_trainer_subclass`
    # below).
    HELPERS_REACHED_INDIRECTLY: ClassVar[tuple[tuple[str, ModuleType], ...]] = (
        ("missing_checkpoint_files", game_train),
        ("trim_micro_batch", game_train),
        ("MicroBatchTrim", game_train),
        ("refuse_empty_trace_overwrite", preflight),
        ("trace_file_path", preflight),
    )

    # The two `PaddingTrimmedGRPOTrainer` overrides: a definition by either name in the reward-hacking
    # trainer is the trace guard or the trim written a second time, whatever class it hangs off.
    TRAINER_OVERRIDE_NAMES: ClassVar[tuple[str, ...]] = ("log", "_prepare_inputs")

    @staticmethod
    def defined_names(module: ModuleType) -> set[str]:
        """Every `def` and `class` name the module's source defines, nested and method names included.

        Parsed rather than pattern-matched: a paren-less dataclass (`class MicroBatchTrim:`) and a
        header split by a magic trailing comma (`class X(\n    GRPOTrainer,\n):`) both survived the
        `def X(` / `class X(` string checks this replaced (review, 2026-09-03), and `ruff format`
        keeps both shapes as they are.
        """
        tree = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }

    @pytest.mark.parametrize(("helper", "home"), [*SHARED_HELPERS, *HELPERS_REACHED_INDIRECTLY])
    def test_neither_trainer_keeps_a_private_copy(self, helper: str, home: ModuleType):
        """A re-added `def helper`, `class helper` or an underscored twin is the drift this removes.

        Each helper is exempt only in its canonical home. `resolve_tokenizer` lives in
        `games.preflight`, so its exemption covers neither trainer: a definition reappearing in
        either one is the drift. The underscored twin is never exempt, because a `_helper` beside
        the real one is how a wrapper starts diverging.
        """
        for module in (rh_train, game_train):
            defined = self.defined_names(module)
            if module is not home:
                assert helper not in defined, (module.__name__, helper)
            assert f"_{helper}" not in defined, (module.__name__, helper)

    def test_the_reward_hacking_trainer_defines_no_trainer_subclass(self):
        """The trainer class is the games one outright, not a subclass restating one behaviour of it.

        `TraceGuardedGRPOTrainer` was that subclass: a second copy of the trace guard's `log`, kept
        because this trainer "had no trim". It takes the trim from the same class now, so any
        `class X(...GRPOTrainer)` here is where a second `log` or a second trim would reappear, and
        the padding trim's neutrality measurement (`test_rh_train_padding_trim`) would no longer be
        about the class a run constructs. Judged on the parsed class bases, and on the two override
        names outright, so neither a formatting shape nor a class with some other base hides one.
        """
        tree = ast.parse(Path(str(rh_train.__file__)).read_text(encoding="utf-8"))
        trainer_subclasses = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any("GRPOTrainer" in ast.unparse(base) for base in node.bases)
        ]
        assert trainer_subclasses == []
        assert self.defined_names(rh_train).isdisjoint(self.TRAINER_OVERRIDE_NAMES)

    @pytest.mark.parametrize(("helper", "home"), list(SHARED_HELPERS))
    def test_the_reward_hacking_trainer_reaches_the_shared_helper(
        self, helper: str, home: ModuleType
    ):
        """The name the reward-hacking trainer calls is the very object its canonical home defines."""
        assert getattr(rh_train, helper) is getattr(home, helper)

    def test_the_shared_json_writer_states_its_encoding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The pair had already drifted: the games copy wrote at the locale's preferred encoding.

        Asserted at the write call itself, not lexically: a substring check on the function's
        source matched ``encoding=`` in its own docstring, so deleting the real argument stayed
        green. And the bytes on disk cannot carry the assertion either -- ``json.dumps`` escapes
        non-ASCII by default, so the written text is pure ASCII under any locale and only the
        encoding argument the call passes distinguishes an explicit-UTF-8 writer from a
        locale-default one.
        """
        seen: dict[str, str | None] = {}
        real_write_text = Path.write_text

        def capturing_write_text(
            path: Path,
            data: str,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> int:
            seen["encoding"] = encoding
            return real_write_text(path, data, encoding=encoding, errors=errors, newline=newline)

        monkeypatch.setattr(Path, "write_text", capturing_write_text)
        target = tmp_path / "nested" / "record.json"
        game_train.write_json(target, {"note": "a non-ASCII character: é"})
        assert seen["encoding"] == "utf-8"
        assert json.loads(target.read_text(encoding="utf-8"))["note"].endswith("é")
