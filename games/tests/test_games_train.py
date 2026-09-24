"""Offline tests for the arm registry, the CLI, the sizing arithmetic and the run artifacts.

No GPU, no network, no model load. Everything that would need one of those -- the tokenizer, the
trainer, the checkpoint config -- is stubbed, which is why `train.py` keeps those reads in small
functions that take what they need rather than reaching for a global.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig
from trl import GRPOConfig  # pyright: ignore[reportPrivateImportUsage]

from games import corpus_partition, generation, preflight, sizing
from games import train as gt
from games.arms import CORPUS_PARTITION_COLUMN, GameArm
from games.payoffs import STATED_RETURN_UNSET
from games.rewards import FRAMING_ID_COLUMN, FRAMING_ID_UNSET, care_grading
from games.s3_sync import S3SyncCallback
from grpo.estimator_defaults import VLLM_IMPORTANCE_SAMPLING_MODE

FIXED_TIMESTAMP = datetime(2026, 8, 17, 12, 30, 0, tzinfo=UTC)
# Named once so the mixed-corpus cases read their game set off the registry instead of repeating it.
BREADTH_ARM = "prosocial-breadth-care1"


def make_config(**overrides: object) -> gt.GameTrainConfig:
    """Build a valid config for tests, overriding whatever the test cares about."""
    base: dict[str, object] = {"arm": "twin-pd-group", "generate_fresh": True}
    return gt.GameTrainConfig(**(base | overrides))  # pyright: ignore[reportArgumentType]


def hybrid_text_config(n_blocks: int = 8) -> SimpleNamespace:
    """A Qwen3.5-shaped config: three linear-attention layers per full-attention layer."""
    return SimpleNamespace(
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        * n_blocks,
        num_key_value_heads=4,
        head_dim=128,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )


def plain_text_config(n_layers: int = 28) -> SimpleNamespace:
    """A Qwen3-shaped config: standard attention only, and no `linear_*` fields at all."""
    return SimpleNamespace(
        layer_types=["full_attention"] * n_layers,
        num_key_value_heads=8,
        head_dim=128,
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


class TestConfigValidation:
    def test_unknown_arm_is_rejected(self):
        with pytest.raises(ValueError, match="unknown arm"):
            gt.GameTrainConfig(arm="not-an-arm", generate_fresh=True)

    def test_an_empty_model_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_id must be non-empty"):
            make_config(model_id="")

    def test_an_empty_explicit_model_source_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_source must be non-empty"):
            make_config(model_source="")

    def test_a_corpus_and_a_fresh_generation_are_different_experiments(self):
        with pytest.raises(ValueError, match="not both"):
            gt.GameTrainConfig(arm="dictator", corpus_path="corpus.jsonl", generate_fresh=True)

    def test_training_without_a_corpus_must_be_asked_for(self):
        with pytest.raises(ValueError, match="no corpus given"):
            gt.GameTrainConfig(arm="dictator")

    def test_a_group_of_one_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2 generations"):
            make_config(num_generations=1)

    def test_greedy_decoding_is_rejected(self):
        with pytest.raises(ValueError, match="temperature must be positive"):
            make_config(temperature=0.0)

    def test_the_vram_fraction_must_be_a_fraction(self):
        with pytest.raises(ValueError, match="vram_usable_fraction"):
            make_config(vram_usable_fraction=1.5)

    def test_requested_episodes_is_the_product_of_group_and_prompts(self):
        assert make_config(num_generations=4, prompts_per_step=6).requested_episodes_per_step == 24


class TestMeasuredCompletionBudget:
    """The directive, encoded: a run we read science from carries a measured budget by default.

    The night this guard came from was spent concluding Qwen3.5 was broken when it was being cut off
    at 2,048 tokens -- a fifth of its median termination length. A warning would have been ignored
    the same way the original 2,048 was chosen, so this refuses.
    """

    def test_the_default_budget_is_the_measured_floor(self):
        assert gt.MEASURED_TERMINATION_BUDGET == 16384
        assert make_config().max_completion_tokens == gt.MEASURED_TERMINATION_BUDGET
        assert gt._parse_args(["--arm", "dictator", "--generate-fresh"]).max_completion_tokens == (
            gt.MEASURED_TERMINATION_BUDGET
        )

    def test_a_thinking_run_below_the_floor_is_refused(self):
        with pytest.raises(ValueError, match="needs at least 16384 completion tokens"):
            make_config(max_completion_tokens=2048)

    def test_the_refusal_names_both_escape_hatches(self):
        with pytest.raises(ValueError, match="allow-short-completions"):
            make_config(max_completion_tokens=8192)
        with pytest.raises(ValueError, match="no-thinking"):
            make_config(max_completion_tokens=8192)

    def test_eight_thousand_is_not_close_enough(self):
        """8,192 sounds generous and reaches only 62% of rollouts on the 4B, 38% on the 2B."""
        with pytest.raises(ValueError, match="format-dominated"):
            make_config(max_completion_tokens=8192)

    def test_an_explicit_override_is_honoured_for_a_timing_probe(self):
        config = make_config(max_completion_tokens=1024, allow_short_completions=True)
        assert config.max_completion_tokens == 1024

    def test_a_plumbing_run_is_exempt_because_it_is_already_labelled(self):
        config = make_config(max_completion_tokens=512, thinking=False)
        assert config.max_completion_tokens == 512
        assert config.thinking is False

    def test_smoke_is_exempt(self):
        """Smoke shrinks the budget to 2,048 on purpose and labels itself plumbing."""
        shrunk = gt.shrink_for_smoke(make_config())
        assert shrunk.max_completion_tokens == gt.SMOKE_COMPLETION_TOKENS
        assert shrunk.smoke is True

    def test_a_budget_above_the_floor_passes(self):
        """A budget above the floor passes.

        Correction OFF because a 32k budget with it ON is refused for memory, not for the floor --
        the refusal under test here is the termination floor alone.
        """
        config = make_config(max_completion_tokens=32768, vllm_importance_sampling_correction=False)
        assert config.max_completion_tokens == 32768


class TestOutputDirNaming:
    def test_a_run_directory_names_the_arm_the_model_and_the_launch(self):
        path = gt.default_output_dir("twin-pd-group", "Qwen/Qwen3.5-4B", timestamp=FIXED_TIMESTAMP)
        assert path == "artifacts/games/runs/twin-pd-group-Qwen-Qwen3.5-4B-20260817-123000"

    def test_a_smoke_run_is_distinguishable_on_disk(self):
        path = gt.default_output_dir(
            "dictator", "Qwen/Qwen3-0.6B", timestamp=FIXED_TIMESTAMP, smoke=True
        )
        assert path.startswith("artifacts/games/runs/smoke-dictator-")

    def test_the_path_safe_model_id_keeps_everything_but_the_separator(self):
        """The path safe model id keeps everything but the separator.

        Named for what it produces, because each sequence plan carries its own `model_slug()` that
        lowercases and strips dots -- a different string for the same input.
        """
        assert gt.path_safe_model_id("Qwen/Qwen3.5-4B") == "Qwen-Qwen3.5-4B"


class TestParseArgs:
    def test_colocate_sleep_offload_is_explicit(self):
        default = gt._parse_args(["--arm", "dictator", "--generate-fresh"])
        requested = gt._parse_args(
            ["--arm", "dictator", "--generate-fresh", "--colocate-sleep-offload"]
        )
        assert default.colocate_sleep_offload is False
        assert requested.colocate_sleep_offload is True

    def test_flags_map_onto_config_fields(self):
        config = gt._parse_args(
            [
                "--arm",
                "iterated-pd-tft",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--corpus",
                "artifacts/games/select/iterated.jsonl",
                "--num-generations",
                "16",
                "--prompts-per-step",
                "4",
                "--max-steps",
                "70",
                "--learning-rate",
                "2e-5",
                "--max-completion-tokens",
                "32768",
                "--leave-one-out",
                "--no-liger",
                "--no-autosize",
                # At 32k the correction must be off wherever colocate defaults on, as it now does.
                "--no-vllm-importance-sampling-correction",
            ]
        )
        assert config.arm == "iterated-pd-tft"
        assert config.model_id == "Qwen/Qwen3.5-2B"
        assert config.corpus_path == "artifacts/games/select/iterated.jsonl"
        assert config.num_generations == 16
        assert config.prompts_per_step == 4
        assert config.max_steps == 70
        assert config.learning_rate == pytest.approx(2e-5)
        assert config.max_completion_tokens == 32768
        assert config.leave_one_out is True
        assert config.use_liger_kernel is False
        assert config.autosize is False
        assert config.smoke is False
        assert config.vllm_importance_sampling_correction is False

    def test_logical_model_identity_is_separate_from_the_immutable_load_source(
        self, tmp_path: Path
    ):
        snapshot = str(tmp_path / "qwen3.5-9b-exact-snapshot")
        config = gt._parse_args(
            [
                "--arm",
                "twin-pd-group",
                "--model",
                "Qwen/Qwen3.5-9B",
                "--model-source",
                snapshot,
                "--generate-fresh",
                "--no-vllm-importance-sampling-correction",
            ]
        )

        assert config.model_id == "Qwen/Qwen3.5-9B"
        assert config.model_source == snapshot
        assert config.load_source == snapshot
        assert config.max_completion_tokens == 32768

    def test_resume_identity_refuses_a_different_immutable_snapshot(self) -> None:
        recorded = make_config(model_source="/snapshots/first")
        current = make_config(model_source="/snapshots/second")

        with pytest.raises(RuntimeError, match="model_source"):
            gt.assert_resume_matches(
                recorded={**gt.RESUME_IDENTITY_DEFAULTS, **asdict(recorded)},
                current=asdict(current),
                fields=gt.RESUME_IDENTITY_FIELDS,
                checkpoint="checkpoint-1",
                consequence="weights changed under one run identity.",
            )

    def test_an_unregistered_arm_is_refused_by_the_parser(self):
        with pytest.raises(SystemExit):
            gt._parse_args(["--arm", "twin-pd-sideways", "--generate-fresh"])

    def test_the_default_model_is_the_working_measurement_size(self):
        config = gt._parse_args(["--arm", "twin-pd-group", "--generate-fresh"])
        assert config.model_id == gt.DEFAULT_MODEL_ID
        assert config.model_id == "Qwen/Qwen3.5-4B"

    def test_the_output_directory_is_derived_when_none_is_given(self):
        config = gt._parse_args(["--arm", "dictator", "--generate-fresh"])
        assert config.output_dir is not None
        assert config.output_dir.startswith(f"{gt.RUN_ROOT}/dictator-Qwen-Qwen3.5-4B-")

    def test_an_explicit_output_directory_survives(self, tmp_path: Path):
        chosen = str(tmp_path / "elsewhere")
        config = gt._parse_args(["--arm", "dictator", "--generate-fresh", "--output-dir", chosen])
        assert config.output_dir == chosen

    def test_training_without_a_corpus_still_needs_the_flag(self):
        with pytest.raises(ValueError, match="no corpus given"):
            gt._parse_args(["--arm", "dictator"])


class TestSmokeShrinkage:
    def test_smoke_shrinks_every_size_knob(self):
        shrunk = gt.shrink_for_smoke(make_config(model_id="Qwen/Qwen3-0.6B"))
        assert shrunk.smoke is True
        assert shrunk.num_generations == 4
        assert shrunk.prompts_per_step == 2
        assert shrunk.max_steps == 3
        assert shrunk.max_prompt_tokens == 512
        assert shrunk.max_completion_tokens == 2048
        assert shrunk.max_prompts == 4
        assert shrunk.logging_steps == 1
        assert shrunk.generate_fresh is True
        assert shrunk.corpus_path is None

    def test_the_smoke_completion_budget_leaves_room_for_a_whole_thought(self):
        """Qwen3-0.6B emits its own <think> block and rambles.

        Measured at the training sampler, 32 samples per budget: 192 tokens parsed 0 of 32 with
        every completion hitting the cap, 1024 parsed 15, 2048 parsed 28. A budget that truncates
        the thinking leaves no visible answer, so every completion is a parse failure and the
        reward's entire-batch raise fails the smoke for a reason unrelated to the plumbing it
        exists to test.
        """
        assert gt.SMOKE_COMPLETION_TOKENS == 2048
        assert gt.SMOKE_OVERRIDES["max_completion_tokens"] == gt.SMOKE_COMPLETION_TOKENS
        assert gt.shrink_for_smoke(make_config()).max_completion_tokens >= 2048

    def test_smoke_still_writes_a_checkpoint(self):
        """Smoke still writes a checkpoint.

        A smoke run that never reaches save_steps would leave the checkpoint path untested, which
        is one of the seams a smoke run exists to exercise.
        """
        shrunk = gt.shrink_for_smoke(make_config())
        assert shrunk.save_steps < shrunk.max_steps

    def test_smoke_leaves_the_experiment_identity_alone(self):
        config = make_config(arm="stag-hunt-group", seed=7, learning_rate=3e-5)
        shrunk = gt.shrink_for_smoke(config)
        assert shrunk.arm == "stag-hunt-group"
        assert shrunk.seed == 7
        assert shrunk.learning_rate == pytest.approx(3e-5)

    def test_the_smoke_flag_defaults_the_model_to_the_plumbing_tier(self):
        config = gt._parse_args(["--arm", "twin-pd-group", "--smoke"])
        assert config.model_id == gt.SMOKE_MODEL_ID
        assert config.smoke is True
        assert config.max_steps == 3
        assert config.output_dir is not None
        assert "/smoke-twin-pd-group-" in config.output_dir

    def test_an_explicit_model_survives_the_smoke_flag(self):
        config = gt._parse_args(["--arm", "twin-pd-group", "--smoke", "--model", "Qwen/Qwen3.5-2B"])
        assert config.model_id == "Qwen/Qwen3.5-2B"
        assert config.max_steps == 3


class TestPrefilledThinkDerivation:
    """Derived by rendering, because the ladder's defaults move non-monotonically by model."""

    @staticmethod
    def tokenizer_rendering(rendered: str) -> SimpleNamespace:
        return SimpleNamespace(apply_chat_template=lambda *_args, **_kwargs: rendered)

    def test_an_unclosed_think_tail_means_the_template_prefills(self):
        """Qwen3.5/3.8 with enable_thinking=True: generation starts inside the thinking block."""
        tokenizer = self.tokenizer_rendering("<|im_start|>assistant\n<think>\n")
        assert preflight.derive_prefilled_think(tokenizer) is True  # pyright: ignore[reportArgumentType]

    def test_a_closed_empty_block_does_not_count_as_prefilled(self):
        """Thinking-off rendering: the prompt carries a complete empty block."""
        tokenizer = self.tokenizer_rendering("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert preflight.derive_prefilled_think(tokenizer) is False  # pyright: ignore[reportArgumentType]

    def test_no_think_tag_at_all_does_not_count_as_prefilled(self):
        """Qwen3-0.6B: the model emits both tags itself."""
        tokenizer = self.tokenizer_rendering("<|im_start|>assistant\n")
        assert preflight.derive_prefilled_think(tokenizer) is False  # pyright: ignore[reportArgumentType]

    def test_a_closed_block_followed_by_a_fresh_open_tag_counts_as_prefilled(self):
        tokenizer = self.tokenizer_rendering(
            "<think>\n\n</think>\n\nprevious turn<|im_start|>assistant\n<think>\n"
        )
        assert preflight.derive_prefilled_think(tokenizer) is True  # pyright: ignore[reportArgumentType]


class TestS3SyncWiring:
    """Artifact shipping is off locally and on for Batch, without a second source of truth."""

    def test_no_destination_means_no_s3_callback(self):
        callbacks = gt.build_callbacks(logging_steps=1, output_dir="/tmp/run", s3_dest="")  # noqa: S108
        assert not any(isinstance(callback, S3SyncCallback) for callback in callbacks)

    def test_a_destination_appends_the_shipper_pointed_at_the_run_directory(self, tmp_path: Path):
        run = str(tmp_path / "run")
        callbacks = gt.build_callbacks(
            logging_steps=1, output_dir=run, s3_dest="s3://bucket/games/twin-pd-group"
        )
        shippers = [c for c in callbacks if isinstance(c, S3SyncCallback)]
        assert len(shippers) == 1
        assert shippers[0].local_dir == Path(run)
        assert shippers[0].s3_dest == "s3://bucket/games/twin-pd-group"

    def test_the_memory_and_reward_callbacks_are_always_present(self):
        callbacks = gt.build_callbacks(logging_steps=1, output_dir="/tmp/run", s3_dest="")  # noqa: S108
        names = {type(callback).__name__ for callback in callbacks}
        assert {
            "MemoryMonitorCallback",
            "RewardLoggingCallback",
            "NonFiniteMetricCallback",
        } <= names

    def test_a_destination_that_is_not_an_s3_uri_fails_at_startup(self):
        """A destination that is not an s3 uri fails at startup.

        build_sync_command would reject it too, but only at the first checkpoint save -- minutes
        into a run that has already paid for a model load.
        """
        with pytest.raises(ValueError, match="must be an s3:// URI"):
            make_config(s3_dest="/local/not/a/bucket")

    def test_the_destination_defaults_from_the_batch_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The destination defaults from the batch environment.

        cloud/submit_job.py --s3-dest sets GAMES_S3_DEST on the container and cloud/entrypoint.sh
        reads the same variable, so a flag-only default would leave the in-run liveness sync off on
        every Batch job.
        """
        monkeypatch.setenv("GAMES_S3_DEST", "s3://bucket/prefix")
        assert gt._parse_args(["--arm", "dictator", "--generate-fresh"]).s3_dest == (
            "s3://bucket/prefix"
        )

    def test_an_explicit_flag_overrides_the_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_S3_DEST", "s3://bucket/from-env")
        config = gt._parse_args(
            ["--arm", "dictator", "--generate-fresh", "--s3-dest", "s3://bucket/from-flag"]
        )
        assert config.s3_dest == "s3://bucket/from-flag"

    def test_a_local_run_ships_nothing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("GAMES_S3_DEST", raising=False)
        assert gt._parse_args(["--arm", "dictator", "--generate-fresh"]).s3_dest == ""


class TestSingleProcessGuard:
    """Multi-process training breaks the reward's group-contiguity assumption silently."""

    def test_a_plain_launch_reads_as_one_process(self):
        assert gt.detected_world_size({}) == 1
        assert gt.detected_world_size({"WORLD_SIZE": "1"}) == 1

    def test_torchrun_and_accelerate_variables_are_both_read(self):
        assert gt.detected_world_size({"WORLD_SIZE": "4"}) == 4
        assert gt.detected_world_size({"LOCAL_WORLD_SIZE": "2"}) == 2
        assert gt.detected_world_size({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "2"}) == 8

    def test_junk_in_the_environment_does_not_read_as_a_process_count(self):
        assert gt.detected_world_size({"WORLD_SIZE": ""}) == 1
        assert gt.detected_world_size({"WORLD_SIZE": "not-a-number"}) == 1

    def test_one_process_passes(self):
        gt.assert_single_process(1, source="test")

    def test_more_than_one_process_is_refused_and_says_where_it_saw_it(self):
        with pytest.raises(RuntimeError, match="single-process training only"):
            gt.assert_single_process(2, source="trainer accelerator")
        with pytest.raises(RuntimeError, match="trainer accelerator"):
            gt.assert_single_process(2, source="trainer accelerator")


class TestTraceCompleteness:
    """The per-completion parquet trace is the project's raw material; holes in it are silent."""

    def test_one_generation_per_step_logged_every_step_is_complete(self):
        assert (
            gt.check_trace_completeness(
                generation_batch_size=64, logging_steps=1, episodes_per_step=64
            )
            is True
        )

    def test_gradient_accumulation_does_not_by_itself_create_holes(self):
        """Gradient accumulation does not by itself create holes.

        TRL sets generation_batch_size = per_device_batch * steps_per_generation, and
        steps_per_generation defaults to gradient_accumulation_steps, so a step split into 8
        accumulation micro-steps still buffers all 64 of its completions. Forcing accumulation to 1
        to "protect" the trace would instead force the micro-batch to the whole episode batch,
        which is what the measured runs OOM on.
        """
        for micro, accum in ((8, 8), (4, 16), (2, 32), (64, 1)):
            assert gt.check_trace_completeness(
                generation_batch_size=micro * accum, logging_steps=1, episodes_per_step=64
            )

    def test_a_buffer_narrower_than_a_step_is_a_structural_break(self):
        with pytest.raises(RuntimeError, match="silently lose completions"):
            gt.check_trace_completeness(
                generation_batch_size=16, logging_steps=1, episodes_per_step=64
            )

    def test_logging_less_often_than_every_step_is_reported_as_incomplete(self):
        assert (
            gt.check_trace_completeness(
                generation_batch_size=64, logging_steps=5, episodes_per_step=64
            )
            is False
        )


class TestChatTemplateKwargs:
    def test_a_template_with_a_reasoning_effort_knob_is_pinned_to_medium(self):
        """A template with a reasoning effort knob is pinned to medium.

        Qwen3.8-27B injects an unauthored system message at its default effort; only `medium`
        renders empty steering text.
        """
        tokenizer = SimpleNamespace(
            get_chat_template=lambda: "{%- if reasoning_effort == 'xhigh' %}...{%- endif %}"
        )
        assert preflight.resolve_chat_template_kwargs(tokenizer) == {"reasoning_effort": "medium"}  # pyright: ignore[reportArgumentType]

    def test_a_template_without_the_knob_gets_no_kwargs(self):
        tokenizer = SimpleNamespace(get_chat_template=lambda: "{%- if enable_thinking %}...")
        assert preflight.resolve_chat_template_kwargs(tokenizer) == {}  # pyright: ignore[reportArgumentType]


class TestSequenceCost:
    def test_a_hybrid_stack_charges_for_its_recurrent_state(self):
        cost = sizing.sequence_cost(hybrid_text_config())
        assert cost.n_linear_attention_layers == 24
        assert cost.n_kv_caching_layers == 8
        assert cost.recurrent_bytes_per_sequence > 0
        assert cost.prefill_upcast_bytes_per_token > 0

    def test_a_plain_attention_stack_has_no_linear_fields_to_read(self):
        """A plain attention stack has no linear fields to read.

        Qwen3-0.6B's config carries none of the `linear_*` keys, so touching them would be an
        AttributeError on the smoke model rather than a number.
        """
        cost = sizing.sequence_cost(plain_text_config())
        assert cost.n_linear_attention_layers == 0
        assert cost.recurrent_bytes_per_sequence == 0
        assert cost.prefill_upcast_bytes_per_token == 0
        assert cost.kv_bytes_per_token > 0

    def test_sliding_window_layers_are_counted_as_caching(self):
        config = plain_text_config(4)
        config.layer_types = [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        assert sizing.sequence_cost(config).n_kv_caching_layers == 4

    def test_a_longer_context_costs_more(self):
        cost = sizing.sequence_cost(hybrid_text_config())
        short = cost.gib_per_episode(prompt_tokens=512, completion_tokens=512)
        long = cost.gib_per_episode(prompt_tokens=2048, completion_tokens=2048)
        assert long > short > 0


class TestPlanSizing:
    def test_a_batch_that_fits_is_left_alone(self):
        plan = make_plan()
        assert plan.clamped is False
        assert plan.num_generations == 8
        assert plan.prompts_per_step == 8
        assert plan.episodes_per_step == 64
        assert plan.gradient_accumulation_steps == 8
        assert plan.micro_batch_size == 8

    def test_prompts_are_clamped_before_the_group_size(self):
        """The group size is the experiment: GRPO's advantage baseline is computed within it."""
        plan = make_plan(free_vram_gib=18.0, weights_gib=8.0)
        assert plan.clamped is True
        assert plan.num_generations == 8
        assert plan.prompts_per_step < 8
        assert plan.episodes_per_step <= plan.episodes_that_fit

    def test_the_group_size_is_clamped_only_when_one_group_will_not_fit(self):
        cost = sizing.sequence_cost(hybrid_text_config())
        per_episode = cost.gib_per_episode(prompt_tokens=1024, completion_tokens=1024)
        # Headroom for four episodes and a bit, against a requested group of eight.
        free = (8.0 + sizing.UNMODELLED_OVERHEAD_GIB + 4.2 * per_episode) / 0.9
        plan = make_plan(free_vram_gib=free, weights_gib=8.0)
        assert plan.clamped is True
        assert plan.num_generations == 4
        assert plan.prompts_per_step == 1
        assert "plumbing rather than as a measurement" in plan.reason

    def test_it_refuses_rather_than_shortening_the_token_budget(self):
        cost = sizing.sequence_cost(hybrid_text_config())
        per_episode = cost.gib_per_episode(prompt_tokens=1024, completion_tokens=1024)
        # Headroom for one episode: below a group of two, so there is no batch left to shrink.
        free = (8.0 + sizing.UNMODELLED_OVERHEAD_GIB + 1.5 * per_episode) / 0.9
        with pytest.raises(RuntimeError, match="never shortened automatically"):
            make_plan(free_vram_gib=free, weights_gib=8.0)

    def test_weights_that_do_not_fit_at_all_are_a_different_message(self):
        with pytest.raises(RuntimeError, match="nothing left for episodes"):
            make_plan(free_vram_gib=10.0, weights_gib=50.0)

    def test_the_failure_message_names_the_model_and_card(self):
        with pytest.raises(RuntimeError, match=re.escape("Qwen3.8-27B on an L4")):
            make_plan(free_vram_gib=10.0, weights_gib=50.0, label="Qwen3.8-27B on an L4")

    def test_autosize_off_keeps_the_requested_batch(self):
        plan = make_plan(free_vram_gib=18.0, weights_gib=8.0, autosize=False)
        assert plan.clamped is False
        assert plan.episodes_per_step == 64
        assert plan.episodes_that_fit < 64

    def test_the_unmodelled_overhead_is_charged_before_any_episode(self):
        """The unmodelled overhead is charged before any episode.

        The config arithmetic accounted for only ~60% of measured peak on this box, so a plan built
        on the arithmetic alone would OOM. See sizing.UNMODELLED_OVERHEAD_GIB.
        """
        generous = make_plan(free_vram_gib=44.0, weights_gib=8.0)
        assert generous.episode_headroom_gib == pytest.approx(
            44.0 * 0.9 - 8.0 - sizing.UNMODELLED_OVERHEAD_GIB
        )

    def test_a_micro_batch_that_does_not_divide_the_batch_is_refused(self):
        with pytest.raises(ValueError, match="must divide"):
            make_plan(num_generations=8, prompts_per_step=2, micro_batch_size=3)

    def test_a_micro_batch_may_be_smaller_than_a_group(self):
        plan = make_plan(num_generations=8, prompts_per_step=2, micro_batch_size=4)
        assert plan.micro_batch_size == 4
        assert plan.gradient_accumulation_steps == 4

    def test_a_smaller_card_yields_a_smaller_plan(self):
        big = make_plan(free_vram_gib=44.0, weights_gib=8.0)
        small = make_plan(free_vram_gib=20.0, weights_gib=8.0)
        assert small.episodes_per_step <= big.episodes_per_step
        assert small.usable_vram_gib < big.usable_vram_gib


class TestSizingAgainstMeasuredRuns:
    """Check the predictor against every configuration actually run on this box.

    `docs/scratch/measured-throughput.md` recorded 16 real Qwen3.5 GRPO configurations on the
    local L4 -- which fitted, which OOM'd, and where. A sizing model that only agrees with itself
    is not a model, and the first version of this one was optimistic enough to plan batches that
    the table says die in prefill. Two independent ceilings have to be reproduced: how many
    episodes generation can hold, and how many tokens one training micro-batch can hold.
    """

    L4_USABLE_GIB = 22.06
    # Parameters as loaded, including the resident vision tower, per that doc's own note.
    WEIGHTS_GIB: ClassVar[dict[str, float]] = {"4B": 8.51, "2B": 4.15}

    # (model, episodes, prompt, completion, micro-batch, did it fit on the L4)
    MEASURED: ClassVar[list[tuple[str, int, int, int, int, bool]]] = [
        ("4B", 32, 2048, 2048, 8, False),
        ("4B", 24, 2048, 2048, 8, False),
        ("4B", 16, 2048, 2048, 8, False),
        ("4B", 16, 2048, 2048, 4, False),
        ("4B", 24, 2048, 2048, 2, True),
        ("4B", 16, 2048, 2048, 2, True),
        ("4B", 16, 512, 512, 8, True),
        ("2B", 16, 512, 512, 8, True),
        ("4B", 16, 512, 128, 8, True),
        ("4B", 16, 512, 32, 8, True),
        ("4B", 8, 1024, 256, 4, True),
        ("4B", 4, 1024, 256, 2, True),
        ("4B", 4, 4096, 256, 2, True),
        ("4B", 8, 4096, 256, 4, False),
        ("4B", 4, 16384, 256, 2, False),
        ("4B", 8, 16384, 256, 4, False),
    ]

    # Inlined so these tests need no network; RLVR_SMOKE re-reads the live config and compares.
    ARCHITECTURES: ClassVar[dict[str, dict[str, int]]] = {
        "4B": {
            "n_linear": 24,
            "n_full": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        },
        "2B": {
            "n_linear": 18,
            "n_full": 6,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        },
    }

    @classmethod
    def cost_for(cls, model: str) -> sizing.SequenceCost:
        fields = cls.ARCHITECTURES[model]
        stub = SimpleNamespace(
            layer_types=["linear_attention"] * fields["n_linear"]
            + ["full_attention"] * fields["n_full"],
            num_key_value_heads=fields["num_key_value_heads"],
            head_dim=fields["head_dim"],
            linear_num_value_heads=fields["linear_num_value_heads"],
            linear_key_head_dim=fields["linear_key_head_dim"],
            linear_value_head_dim=fields["linear_value_head_dim"],
        )
        return sizing.sequence_cost(stub)

    @pytest.mark.skipif(
        not os.environ.get("RLVR_SMOKE"), reason="reads the real checkpoint config (needs cache)"
    )
    @pytest.mark.parametrize("model", ["4B", "2B"])
    def test_the_inlined_architecture_still_matches_the_checkpoint(self, model: str):
        live = sizing.sequence_cost(
            AutoConfig.from_pretrained(f"Qwen/Qwen3.5-{model}").get_text_config()
        )
        assert self.cost_for(model) == live

    def predict_fits(self, model: str, episodes: int, prompt: int, completion: int, micro: int):
        """Predict a configuration's fate the way plan_sizing decides it."""
        weights = self.WEIGHTS_GIB[model]
        headroom = self.L4_USABLE_GIB * 0.9 - weights - sizing.UNMODELLED_OVERHEAD_GIB
        per_episode = self.cost_for(model).gib_per_episode(
            prompt_tokens=prompt, completion_tokens=completion
        )
        episodes_fit = int(headroom // per_episode)
        budget = sizing.micro_batch_token_budget(
            free_vram_gib=self.L4_USABLE_GIB, weights_gib=weights
        )
        micro_cap = max(budget // (prompt + completion), 1)
        return episodes <= episodes_fit and micro <= micro_cap

    @pytest.mark.parametrize(
        "case", MEASURED, ids=lambda case: "-".join(str(part) for part in case)
    )
    def test_the_predictor_agrees_with_the_measured_outcome(
        self, case: tuple[str, int, int, int, int, bool]
    ):
        model, episodes, prompt, completion, micro, fitted = case
        predicted = self.predict_fits(model, episodes, prompt, completion, micro)
        assert predicted == fitted, (
            f"{model} {episodes}ep {prompt}+{completion} micro={micro}: "
            f"measured fitted={fitted}, predicted fits={predicted}"
        )

    def test_the_measured_episode_ceiling_is_reproduced(self):
        """The measured episode ceiling is reproduced.

        The doc's headline ceiling: 24 episodes at 2048+2048 on the L4, not the 64 the budget note
        assumed.
        """
        assert self.predict_fits("4B", 24, 2048, 2048, 2)
        assert not self.predict_fits("4B", 32, 2048, 2048, 2)

    def test_the_measured_micro_batch_ceiling_is_reproduced(self):
        """The measured micro batch ceiling is reproduced.

        Same episode count, only the micro-batch differing, is what made the reference profile
        runnable at all -- so a plan can be right about episodes and still OOM.
        """
        assert self.predict_fits("4B", 16, 2048, 2048, 2)
        assert not self.predict_fits("4B", 16, 2048, 2048, 4)

    def test_a_bigger_card_affords_a_bigger_micro_batch(self):
        l4 = sizing.micro_batch_token_budget(free_vram_gib=22.06, weights_gib=8.51)
        l40s = sizing.micro_batch_token_budget(free_vram_gib=44.4, weights_gib=8.51)
        assert l40s > l4
        # The 64-episode reference profile should fit an L40S: micro-batch 8 at 1024+1024.
        assert l40s // 2048 >= 8

    def test_the_derived_micro_batch_matches_the_measured_reference_profile(self):
        """4B at 2048+2048 on the L4 with a group of 8: the measured answer is 2."""
        budget = sizing.micro_batch_token_budget(free_vram_gib=22.06, weights_gib=8.51)
        assert (
            sizing._derive_micro_batch(
                num_generations=8, tokens_per_sequence=4096, token_budget=budget
            )
            == 2
        )

    def test_the_derived_micro_batch_always_divides_the_group(self):
        for tokens in (256, 1024, 2048, 4096, 32768):
            micro = sizing._derive_micro_batch(
                num_generations=8, tokens_per_sequence=tokens, token_budget=8704
            )
            assert 8 % micro == 0, tokens
            assert micro >= 1


class TestCorpusLoading:
    @staticmethod
    def write_corpus(path: Path, rows: list[dict[str, Any]]) -> str:
        path.write_text("\n".join(json.dumps(row) for row in rows))
        return str(path)

    @staticmethod
    def row(**overrides: Any) -> dict[str, Any]:
        base = {
            "prompt": "a sheet of paper",
            "prompt_id": "twin-pd-0001",
            "game_id": "twin-pd",
            "grading": "group-mix",
            "payoff_variant": "temptation-2",
        }
        return base | overrides

    def test_a_matching_corpus_loads(self, tmp_path: Path):
        path = self.write_corpus(tmp_path / "corpus.jsonl", [self.row(), self.row()])
        rows = gt.load_corpus(path, gt.ARMS["twin-pd-group"])
        assert len(rows) == 2

    def test_a_corpus_graded_differently_is_refused(self, tmp_path: Path):
        """A corpus graded differently is refused.

        Training twin-pd-self on a group-mix corpus would run, produce numbers, and answer a
        question nobody asked. The two arms differ in nothing else, so nothing else would catch the
        mix-up.
        """
        path = self.write_corpus(tmp_path / "corpus.jsonl", [self.row()])
        with pytest.raises(ValueError, match="but this arm trains"):
            gt.load_corpus(path, gt.ARMS["twin-pd-self"])

    def test_a_corpus_for_another_game_is_refused(self, tmp_path: Path):
        path = self.write_corpus(tmp_path / "corpus.jsonl", [self.row(game_id="stag-hunt")])
        with pytest.raises(ValueError, match="but this arm trains"):
            gt.load_corpus(path, gt.ARMS["twin-pd-group"])

    def test_rows_missing_the_arm_columns_are_refused(self, tmp_path: Path):
        path = self.write_corpus(tmp_path / "corpus.jsonl", [{"prompt": "no schema here"}])
        with pytest.raises(ValueError, match="lack"):
            gt.load_corpus(path, gt.ARMS["twin-pd-group"])

    def test_an_empty_corpus_is_refused(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        path.write_text("\n\n")
        with pytest.raises(ValueError, match="no rows"):
            gt.load_corpus(str(path), gt.ARMS["twin-pd-group"])

    def test_blank_lines_are_skipped(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        path.write_text(json.dumps(self.row()) + "\n\n" + json.dumps(self.row()) + "\n")
        assert len(gt.load_corpus(str(path), gt.ARMS["twin-pd-group"])) == 2

    def breadth_rows(self) -> list[dict[str, Any]]:
        """One row per game of the breadth arm's own set, under its own grading."""
        care = gt.ARMS[BREADTH_ARM].grading
        return [
            self.row(prompt_id=f"{game_id}-0001", game_id=game_id, grading=care)
            for game_id in gt.arm_game_ids(gt.ARMS[BREADTH_ARM])
        ]

    def test_a_mixed_corpus_of_the_arms_games_loads(self, tmp_path: Path):
        """A mixed corpus of the arms games loads.

        The wave-4b corpus: six games under one grading in one file, which every arm before it
        would have refused wholesale.
        """
        path = self.write_corpus(tmp_path / "corpus.jsonl", self.breadth_rows())
        rows = gt.load_corpus(path, gt.ARMS[BREADTH_ARM])
        assert {row["game_id"] for row in rows} == set(gt.arm_game_ids(gt.ARMS[BREADTH_ARM]))

    def test_a_mixed_corpus_carrying_a_game_outside_the_set_is_refused(self, tmp_path: Path):
        """The sabotage case, kept as a test: one row of a game the arm never named.

        Loading it would train prompts the arm's own record does not describe, and the pooled
        cooperation rate would absorb a game whose cooperative action means something else.
        """
        rows = [
            *self.breadth_rows(),
            self.row(prompt_id="hi-lo-0001", game_id="hi-lo", grading=gt.ARMS[BREADTH_ARM].grading),
        ]
        path = self.write_corpus(tmp_path / "corpus.jsonl", rows)
        with pytest.raises(ValueError, match="but this arm trains"):
            gt.load_corpus(path, gt.ARMS[BREADTH_ARM])

    def test_the_refusal_names_the_whole_game_set(self, tmp_path: Path):
        path = self.write_corpus(
            tmp_path / "corpus.jsonl",
            [self.row(game_id="hi-lo", grading=gt.ARMS[BREADTH_ARM].grading)],
        )
        with pytest.raises(ValueError, match="stag-hunt") as raised:
            gt.load_corpus(path, gt.ARMS[BREADTH_ARM])
        assert "hi-lo" in str(raised.value)

    def test_a_single_game_arm_still_refuses_another_of_its_gradings_games(self, tmp_path: Path):
        """A single game arm still refuses another of its gradings games.

        The widening is per arm: an arm that named no extra games keeps refusing everything but its
        own, so the breadth arm's set cannot leak into the arms it was added beside.
        """
        path = self.write_corpus(
            tmp_path / "corpus.jsonl", [self.row(game_id="stag-hunt", grading="group-mix")]
        )
        with pytest.raises(ValueError, match="but this arm trains"):
            gt.load_corpus(path, gt.ARMS["twin-pd-group"])


class TestWhatAMixedCorpusRunRecordsAboutItself:
    """A breadth run's record has to name its own axes, because nothing else can reconstruct them.

    Two facts about a mixed corpus are unrecoverable after the fact from the run directory alone: the
    game set the arm allowed (the registry can be edited between the launch and any later reading) and
    which counterpart framings the corpus actually trained. The second one carries a claim other
    readouts make: the battery's framing table says a framing was never trained, and on this arm that
    is true per run rather than by construction, so the run states it.
    """

    @staticmethod
    def rows(framings: list[str]) -> list[dict[str, Any]]:
        """One row per framing, alternating between two of the arm's games."""
        games = ["twin-pd", "stag-hunt"]
        return [
            {
                "prompt_id": f"prompt-{index}",
                "game_id": games[index % len(games)],
                "grading": gt.ARMS[BREADTH_ARM].grading,
                "payoff_variant": "temptation-2",
                FRAMING_ID_COLUMN: framing,
            }
            for index, framing in enumerate(framings)
        ]

    def composition(self, framings: list[str]) -> dict[str, list[str]]:
        return gt.describe_row_composition(self.rows(framings))

    def test_the_game_and_the_framing_are_both_composition_axes(self):
        assert "game_id" in gt.ROW_COMPOSITION_COLUMNS
        assert FRAMING_ID_COLUMN in gt.ROW_COMPOSITION_COLUMNS
        composition = self.composition(["twin", "human"])
        assert composition["game_id"] == ["stag-hunt", "twin-pd"]
        assert composition[FRAMING_ID_COLUMN] == ["human", "twin"]

    def test_a_corpus_predating_the_framing_column_still_describes_itself(self):
        """A corpus predating the framing column still describes itself.

        Every banked corpus: no framing column at all, and the composition names the axes it has.
        """
        rows = [
            {
                "prompt_id": "prompt-0",
                "game_id": "twin-pd",
                "grading": "group-mix",
                "payoff_variant": "temptation-2",
            }
        ]
        composition = gt.describe_row_composition(rows)
        assert composition["game_id"] == ["twin-pd"]
        assert FRAMING_ID_COLUMN not in composition

    def test_the_unframed_marker_is_described_as_itself_and_trains_no_framing(self):
        """The unframed marker is described as itself and trains no framing.

        The trust rows carry the marker, so the composition shows it (the corpus really does hold
        rows with no framing) while the trained-framings list does not (there is no such framing).
        """
        composition = self.composition(["twin", FRAMING_ID_UNSET])
        assert composition[FRAMING_ID_COLUMN] == [FRAMING_ID_UNSET, "twin"]
        assert gt.framings_trained(composition) == ["twin"]

    def test_the_record_names_the_arms_game_set_and_the_corpus_framings(self, tmp_path: Path):
        payload = gt.run_config_payload(
            make_config(arm=BREADTH_ARM, output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={"corpus_composition": self.composition(["human", "twin", "twin"])},
        )
        assert payload["game_ids"] == list(gt.ARMS[BREADTH_ARM].game_ids)
        assert payload["trained_framing_ids"] == ["human", "twin"]

    def test_a_single_game_arm_records_an_empty_set_and_no_framings(self, tmp_path: Path):
        payload = gt.run_config_payload(
            make_config(output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={"corpus_composition": {"payoff_variant": ["temptation-2"]}},
        )
        assert payload["game_ids"] == []
        assert payload["trained_framing_ids"] == []

    def test_both_fields_survive_the_json_round_trip_as_lists(self, tmp_path: Path):
        payload = gt.run_config_payload(
            make_config(arm=BREADTH_ARM, output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={"corpus_composition": self.composition(["twin"])},
        )
        restored = json.loads(json.dumps(payload, indent=2, default=str))
        assert restored["game_ids"] == list(gt.ARMS[BREADTH_ARM].game_ids)
        assert restored["trained_framing_ids"] == ["twin"]

    def test_the_game_set_joins_the_resume_identity(self, tmp_path: Path):
        """The game set joins the resume identity.

        A resume under a different game set is a different experiment, and the arm NAME cannot
        catch it: a rented box ships the current tree, so the registry entry can gain or lose a
        game between the launch and the relaunch while the name and every other recorded field
        agree.
        """
        recorded = gt.run_config_payload(
            make_config(arm=BREADTH_ARM, output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )
        assert "game_ids" in gt.RESUME_ARM_IDENTITY_FIELDS
        with pytest.raises(RuntimeError, match="game_ids"):
            gt.assert_resume_matches(
                recorded={**gt.RESUME_ARM_IDENTITY_DEFAULTS, **recorded},
                current=gt.arm_identity(gt.ARMS["twin-pd-group"]),
                fields=gt.RESUME_ARM_IDENTITY_FIELDS,
                checkpoint="checkpoint-70",
                consequence="two corpora under one set of step numbers.",
            )

    def test_a_resume_on_the_same_game_set_is_allowed(self, tmp_path: Path):
        recorded = gt.run_config_payload(
            make_config(arm=BREADTH_ARM, output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )
        gt.assert_resume_matches(
            recorded={**gt.RESUME_ARM_IDENTITY_DEFAULTS, **recorded},
            current=gt.arm_identity(gt.ARMS[BREADTH_ARM]),
            fields=gt.RESUME_ARM_IDENTITY_FIELDS,
            checkpoint="checkpoint-70",
            consequence="two corpora under one set of step numbers.",
        )

    def test_a_record_predating_the_field_resumes_as_a_single_game_run(self):
        """A record predating the field resumes as a single game run.

        The precedent is `RESUME_IDENTITY_DEFAULTS`: absence in an old record means the value in
        force when it was written, which for every banked arm is no extra games -- never today's.
        """
        assert gt.RESUME_ARM_IDENTITY_DEFAULTS["game_ids"] == []
        gt.assert_resume_matches(
            recorded={**gt.RESUME_ARM_IDENTITY_DEFAULTS, "arm": "twin-pd-self"},
            current=gt.arm_identity(gt.ARMS["twin-pd-self"]),
            fields=gt.RESUME_ARM_IDENTITY_FIELDS,
            checkpoint="checkpoint-70",
            consequence="two corpora under one set of step numbers.",
        )
        with pytest.raises(RuntimeError, match="game_ids"):
            gt.assert_resume_matches(
                recorded={**gt.RESUME_ARM_IDENTITY_DEFAULTS, "arm": BREADTH_ARM},
                current=gt.arm_identity(gt.ARMS[BREADTH_ARM]),
                fields=gt.RESUME_ARM_IDENTITY_FIELDS,
                checkpoint="checkpoint-70",
                consequence="two corpora under one set of step numbers.",
            )


class TestACorpusCarriesTheModelItWasSelectedFor:
    """`load_corpus` matched the arm but not the model, so a cross-model corpus trained silently.

    A selected corpus is not neutral material: `games.select_prompts` keeps only prompts whose
    action distribution was mixed at training temperature for one specific checkpoint, and for a
    vs-fixed-mix arm every row also carries `opp_coop_prob`, a cached measurement of one specific
    frozen opponent. Training a 9B on a 2B-selected corpus passes every other check in this module;
    the symptom is a raised `frac_groups_pure` that reads as a property of the model rather than of
    the corpus, and for the vs-frozen arms an opponent distribution never sampled from the opponent
    the arm claims.
    """

    @staticmethod
    def rows(model_id: str | None) -> list[dict[str, Any]]:
        row: dict[str, Any] = {
            "prompt": "a sheet of paper",
            "prompt_id": "twin-pd-0001",
            "game_id": "twin-pd",
            "grading": "group-mix",
            "payoff_variant": "temptation-2",
        }
        if model_id is not None:
            row[gt.SELECTED_FOR_MODEL_COLUMN] = model_id
        return [row, dict(row)]

    def test_a_corpus_selected_for_another_model_is_refused(self):
        with pytest.raises(ValueError, match="selected for"):
            gt.assert_corpus_selected_for_model(
                self.rows("Qwen/Qwen3.5-2B"), path="corpus.jsonl", model_id="Qwen/Qwen3.5-9B"
            )

    def test_the_refusal_names_both_models(self):
        with pytest.raises(ValueError, match=r"Qwen3\.5-2B") as raised:
            gt.assert_corpus_selected_for_model(
                self.rows("Qwen/Qwen3.5-2B"), path="corpus.jsonl", model_id="Qwen/Qwen3.5-9B"
            )
        assert "Qwen3.5-9B" in str(raised.value)

    def test_a_matching_corpus_passes(self):
        gt.assert_corpus_selected_for_model(
            self.rows("Qwen/Qwen3.5-4B"), path="corpus.jsonl", model_id="Qwen/Qwen3.5-4B"
        )

    def test_an_unstamped_corpus_says_the_pairing_is_unverified(
        self, caplog: pytest.LogCaptureFixture
    ):
        """An unstamped corpus says the pairing is unverified.

        Corpora written before the column existed carry the model only in their filename, so the
        pairing is convention. That is worth a loud line rather than silence, since the filename is
        the only thing a later reader can check it against.
        """
        with caplog.at_level(logging.WARNING, logger="games.train"):
            gt.assert_corpus_selected_for_model(
                self.rows(None), path="corpus-twin-pd-2B.jsonl", model_id="Qwen/Qwen3.5-9B"
            )
        assert "cannot be checked" in caplog.text
        assert "Qwen3.5-9B" in caplog.text

    def test_the_check_runs_on_the_corpus_path_of_prepare_rows(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in self.rows("Qwen/Qwen3.5-2B")))
        config = make_config(
            arm="twin-pd-group",
            generate_fresh=False,
            corpus_path=str(path),
            model_id="Qwen/Qwen3.5-4B",
        )
        with pytest.raises(ValueError, match="selected for"):
            gt.prepare_rows(config)


class TestPayoffVariantPin:
    @staticmethod
    def rows() -> list[dict[str, Any]]:
        return [
            {"prompt_id": "a", "payoff_variant": "temptation-2"},
            {"prompt_id": "b", "payoff_variant": "temptation-10"},
            {"prompt_id": "c", "payoff_variant": "temptation-2"},
        ]

    def test_an_unpinned_arm_keeps_every_variant(self):
        assert len(gt.filter_payoff_variants(self.rows(), ())) == 3

    def test_the_pin_drops_the_other_magnitudes(self):
        kept = gt.filter_payoff_variants(self.rows(), ("temptation-2",))
        assert [row["prompt_id"] for row in kept] == ["a", "c"]

    def test_a_pin_that_empties_the_corpus_raises(self):
        rows = [{"payoff_variant": "temptation-10"}]
        with pytest.raises(ValueError, match="left no rows"):
            gt.filter_payoff_variants(rows, ("temptation-2",))

    def test_a_pin_with_no_column_to_filter_on_raises(self):
        with pytest.raises(ValueError, match="cannot be enforced"):
            gt.filter_payoff_variants([{"prompt_id": "a"}], ("temptation-2",))


class TestCorpusPartitionPin:
    """The narrowing a mix-split arm rides on: one stamped corpus, two arms, one side each."""

    @staticmethod
    def rows() -> list[dict[str, Any]]:
        return [
            {"prompt_id": "a", "corpus_partition": "above-threshold"},
            {"prompt_id": "b", "corpus_partition": "below-threshold"},
            {"prompt_id": "c", "corpus_partition": "straddling-pair"},
            {"prompt_id": "d", "corpus_partition": "above-threshold"},
        ]

    def test_an_unpinned_arm_keeps_every_row_including_the_straddlers(self):
        assert len(gt.filter_corpus_partition(self.rows(), "")) == 4

    def test_the_pin_keeps_one_side_and_drops_the_straddling_pairs(self):
        kept = gt.filter_corpus_partition(self.rows(), "above-threshold")
        assert [row["prompt_id"] for row in kept] == ["a", "d"]

    def test_a_pin_on_an_unpartitioned_corpus_raises(self):
        """A pin on an unpartitioned corpus raises.

        What a freshly generated corpus looks like: which side of the boundary a prompt sits on is
        a measurement of the base policy, so it cannot be inferred from the prompt.
        """
        with pytest.raises(ValueError, match="cannot be enforced"):
            gt.filter_corpus_partition([{"prompt_id": "a"}], "above-threshold")

    def test_a_pin_that_matches_nothing_raises(self):
        rows = [{"prompt_id": "a", "corpus_partition": "straddling-pair"}]
        with pytest.raises(ValueError, match="left no rows"):
            gt.filter_corpus_partition(rows, "above-threshold")

    def test_a_pin_that_narrows_nothing_raises(self):
        """A pin that narrows nothing raises.

        The quiet failure: a corpus pre-split into one file per side would let both arms load a
        file this pin cannot distinguish from, and the contrast would never have been set up.
        """
        rows = [
            {"prompt_id": "a", "corpus_partition": "above-threshold"},
            {"prompt_id": "b", "corpus_partition": "above-threshold"},
        ]
        with pytest.raises(ValueError, match="narrows nothing"):
            gt.filter_corpus_partition(rows, "above-threshold")

    def test_the_pin_runs_inside_prepare_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The pin runs inside prepare rows.

        Registered through the registry the trainer reads rather than by faking the arm, because
        `GameTrainConfig.game_arm` resolves the name against it. No mix-split arm is registered for
        real yet: on the measured safe-hunt sweep only one pair lands above the boundary.
        """
        rows = [
            {
                "prompt_id": f"stag-{index}",
                "game_id": "stag-hunt",
                "grading": "group-mix",
                "payoff_variant": "safe-hunt",
                "corpus_partition": "above-threshold" if index % 2 else "straddling-pair",
            }
            for index in range(4)
        ]
        path = tmp_path / "corpus.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        probe = replace(
            gt.ARMS["stag-hunt-safe-rung"],
            corpus_partition="above-threshold",
            notes="probe: the above side of the safe rung's boundary",
        )
        monkeypatch.setitem(gt.ARMS, "probe-mix-split", probe)
        config = make_config(arm="probe-mix-split", generate_fresh=False, corpus_path=str(path))
        prepared = gt.prepare_rows(config)
        assert [row["prompt_id"] for row in prepared] == ["stag-1", "stag-3"]

    def test_the_partition_is_named_in_what_a_run_records_about_its_corpus(self):
        """The partition is named in what a run records about its corpus.

        Free instrumentation: without the column here a run's own record could not say which side
        of the split it trained, and the two arms' artifacts would be indistinguishable.
        """
        assert CORPUS_PARTITION_COLUMN in gt.ROW_COMPOSITION_COLUMNS
        composition = gt.describe_row_composition(self.rows())
        assert composition[CORPUS_PARTITION_COLUMN] == [
            "above-threshold",
            "below-threshold",
            "straddling-pair",
        ]

    def test_the_partitioners_group_size_default_matches_the_trainers(self):
        """The partitioners group size default matches the trainers.

        The boundary moves with the group size under leave-one-out, so a partitioner defaulting to
        a different group from the trainer would sort prompts by a number no run ever used.
        """
        assert gt.GameTrainConfig.num_generations == corpus_partition.DEFAULT_NUM_GENERATIONS


class TestPrepareRows:
    def test_a_corpus_path_is_read_filtered_and_capped(self, tmp_path: Path):
        rows = [
            {
                "prompt_id": f"twin-pd-{index}",
                "game_id": "twin-pd",
                "grading": "group-mix",
                "payoff_variant": "temptation-2" if index % 2 else "temptation-10",
            }
            for index in range(6)
        ]
        path = tmp_path / "corpus.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        config = make_config(
            arm="twin-pd-group", generate_fresh=False, corpus_path=str(path), max_prompts=2
        )
        prepared = gt.prepare_rows(config)
        assert len(prepared) == 2

    def test_the_iterated_arm_drops_the_high_temptation_rows(self, tmp_path: Path):
        rows = [
            {
                "prompt_id": f"iter-{index}",
                "game_id": "iterated-pd-tft",
                "grading": "iterated-return",
                "payoff_variant": "temptation-2" if index < 2 else "temptation-10",
            }
            for index in range(5)
        ]
        path = tmp_path / "corpus.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        config = make_config(arm="iterated-pd-tft", generate_fresh=False, corpus_path=str(path))
        prepared = gt.prepare_rows(config)
        assert len(prepared) == 2
        assert {row["payoff_variant"] for row in prepared} == {"temptation-2"}


class FakeParameter:
    """Enough of a torch parameter for the LoRA coverage count."""

    def __init__(self, count: int, *, trainable: bool) -> None:
        self.count = count
        self.requires_grad = trainable

    def numel(self) -> int:
        return self.count


def fake_trainer(module_names: list[str], parameters: list[FakeParameter]) -> SimpleNamespace:
    """A stand-in for a built GRPOTrainer, for the coverage checks only."""
    model = SimpleNamespace(
        named_modules=lambda: [(name, object()) for name in module_names],
        parameters=lambda: parameters,
    )
    return SimpleNamespace(model=model)


class TestLoraCoverage:
    parameters: ClassVar[list[FakeParameter]] = [
        FakeParameter(100, trainable=True),
        FakeParameter(900, trainable=False),
    ]

    def test_a_plain_attention_stack_passes_without_a_linear_attention_adapter(self):
        """A plain attention stack passes without a linear attention adapter.

        grpo.throughput.summarize_lora demands a linear_attn adapter unconditionally, which is
        right on Qwen3.5 and would fail every smoke run on Qwen3-0.6B.
        """
        trainer = fake_trainer(["model.layers.0.self_attn.q_proj.lora_A.default"], self.parameters)
        summary = gt.summarize_lora_for_architecture(
            trainer,  # pyright: ignore[reportArgumentType]
            expected_linear_attention_layers=0,
        )
        assert summary["adapted_modules"] == 1
        assert summary["adapted_linear_attention_modules"] == 0
        assert summary["trainable_fraction"] == pytest.approx(0.1)

    def test_a_hybrid_stack_with_no_linear_attention_adapter_still_raises(self):
        trainer = fake_trainer(["model.layers.0.self_attn.q_proj.lora_A.default"], self.parameters)
        with pytest.raises(RuntimeError, match="linear_attention layer"):
            gt.summarize_lora_for_architecture(
                trainer,  # pyright: ignore[reportArgumentType]
                expected_linear_attention_layers=48,
            )

    def test_a_hybrid_stack_with_the_adapter_present_passes(self):
        trainer = fake_trainer(
            [
                "model.layers.0.linear_attn.in_proj_qkv.lora_A.default",
                "model.layers.3.self_attn.q_proj.lora_A.default",
            ],
            self.parameters,
        )
        summary = gt.summarize_lora_for_architecture(
            trainer,  # pyright: ignore[reportArgumentType]
            expected_linear_attention_layers=48,
        )
        assert summary["adapted_modules"] == 2
        assert summary["adapted_linear_attention_modules"] == 1

    def test_an_unadapted_model_raises_whatever_the_architecture(self):
        trainer = fake_trainer(["model.layers.0.self_attn.q_proj"], self.parameters)
        with pytest.raises(RuntimeError, match="no LoRA adapter landed"):
            gt.summarize_lora_for_architecture(
                trainer,  # pyright: ignore[reportArgumentType]
                expected_linear_attention_layers=0,
            )


def fake_metric_trainer(history: list[dict[str, object]]) -> SimpleNamespace:
    """A stand-in trainer carrying only the log history the read-back reads."""
    return SimpleNamespace(state=SimpleNamespace(log_history=history, global_step=len(history)))


def healthy_record(step: int, **overrides: object) -> dict[str, object]:
    """One log record with every metric a group-mix games run claims to produce."""
    record: dict[str, object] = {
        "step": step,
        "reward": 0.5,
        "reward_std": 0.1,
        "frac_reward_zero_std": 0.25,
        "parse_failure_rate": 0.02,
        "truncated_thinking_rate": 0.01,
        "frac_groups_pure": 0.125,
        "coop_rate": 0.4,
    }
    return record | overrides


def read_back(
    history: list[dict[str, object]],
    *,
    grading: str = "group-mix",
    parse_penalty_mode: str = gt.PARSE_PENALTY_MARGIN_BELOW_WORSE,
) -> tuple[dict[str, object], list[str]]:
    """Read a history back the way a run of this grading would, so the gate under test is real.

    The row-relative mode is the default because it exempts the least: a case about a constant-mode
    run names that mode, and nothing else silently gets a wider exemption than it asked for.
    """
    return gt.read_back_metrics(
        fake_metric_trainer(history),  # pyright: ignore[reportArgumentType]
        required=gt.required_metrics_for(grading),
        constant_by_construction=gt.constant_by_construction_metrics(parse_penalty_mode),
    )


def guard_series_a_constant_mode_run_logs(step: int) -> dict[str, object]:
    """One step of the parse-price guard's series as the constant penalty mode pins them.

    Both denominators are zero because that mode prices from a run knob rather than from a row, so
    there is no answer space to check anything against, and the realised mean is that knob. Only the
    failure count moves, which is what makes it the control in the pair of cases below.
    """
    return {
        "parse_price/n_failures": float(step),
        "parse_price/n_checked": 0.0,
        "parse_price/n_identity_checked": 0.0,
        "parse_price/realised_mean": -1.0,
    }


class TestMetricReadBack:
    def test_a_complete_history_reports_final_and_mean(self):
        summary, missing = read_back([healthy_record(1, reward=0.2), healthy_record(2, reward=0.6)])
        assert missing == []
        assert summary["reward_final"] == pytest.approx(0.6)
        assert summary["reward_mean"] == pytest.approx(0.4)
        assert summary["coop_rate_final"] == pytest.approx(0.4)
        assert summary["n_logged_steps"] == 2

    def test_a_metric_that_never_logged_is_reported_not_silently_dropped(self):
        record = healthy_record(1)
        del record["parse_failure_rate"]
        _, missing = read_back([record])
        assert missing == ["parse_failure_rate"]


class TestTheBehaviouralMetricIsRequiredPerArm:
    """Each arm's ONE behavioural number is inside the read-back gate, not merely optional.

    Both used to be optional, which meant the dictator arm -- whose only behavioural readout is
    the keep fraction, since it has no opponent and therefore no cooperation rate -- could finish,
    write a clean-looking `train_summary.json` with no behavioural number in it, and raise nothing.
    That is the same shape as the dead-metric bug the whole read-back exists for.
    """

    def test_a_keep_fraction_arm_must_have_recorded_its_keep_fraction(self):
        record = healthy_record(1, mean_keep_fraction=0.6)
        del record["mean_keep_fraction"]
        _, missing = read_back([record], grading="keep-fraction")
        assert missing == ["mean_keep_fraction"]

    def test_a_keep_fraction_arm_does_not_need_a_cooperation_rate(self):
        record = healthy_record(1, mean_keep_fraction=0.6)
        del record["coop_rate"]
        summary, missing = read_back([record], grading="keep-fraction")
        assert missing == []
        assert summary["mean_keep_fraction_final"] == pytest.approx(0.6)

    def test_a_group_mix_arm_must_have_recorded_its_cooperation_rate(self):
        record = healthy_record(1)
        del record["coop_rate"]
        _, missing = read_back([record], grading="group-mix")
        assert missing == ["coop_rate"]

    def test_a_group_mix_arm_does_not_need_a_keep_fraction(self):
        summary, missing = read_back([healthy_record(1)], grading="group-mix")
        assert missing == []
        assert "mean_keep_fraction_final" not in summary

    def test_the_keep_fraction_still_reaches_the_summary_of_an_arm_that_does_not_require_it(self):
        """The keep fraction still reaches the summary of an arm that does not require it.

        Optional, so a stray value is still aggregated and still checked for being constant -- what
        changed is which arm's absence raises.
        """
        summary, _ = read_back([healthy_record(1, mean_keep_fraction=0.6)], grading="group-mix")
        assert summary["mean_keep_fraction_final"] == pytest.approx(0.6)

    def test_the_leave_one_out_prior_rate_is_recorded_when_it_is_logged(self):
        """The leave one out prior rate is recorded when it is logged.

        The flag that a reward was computed from UNINFORMATIVE_COOP_PRIOR rather than from data.
        """
        summary, _ = read_back([healthy_record(1, leave_one_out_prior_rate=0.25)])
        assert summary["leave_one_out_prior_rate_final"] == pytest.approx(0.25)
        assert "leave_one_out_prior_rate" in gt.OPTIONAL_METRICS

    def test_a_history_with_no_reward_at_all_raises(self):
        with pytest.raises(RuntimeError, match="nothing was measured"):
            read_back([{"loss": 0.3}])

    def test_slashes_in_trl_metric_names_are_flattened_for_json(self):
        summary, _ = read_back([healthy_record(1, **{"completions/mean_length": 300.0})])
        assert summary["completions_mean_length_final"] == pytest.approx(300.0)


class TestDeadMetricDetection:
    """A metric that never moves satisfies a presence check while measuring nothing.

    This is how TRL's `frac_reward_zero_std` survived three 70-step arms at exactly 0.0000 and got
    cited as evidence that every group disagreed. Presence is not measurement.
    """

    def test_a_metric_pinned_across_the_run_is_reported_as_constant(self):
        history = [healthy_record(step, reward=0.1 * step) for step in range(1, 6)]
        summary, missing = read_back(history)
        assert missing == []
        # frac_reward_zero_std is 0.25 in every healthy_record, i.e. present and pinned.
        constant = cast("list[str]", summary["constant_metrics"])
        assert "frac_reward_zero_std" in constant
        assert "reward" not in constant

    def test_a_varying_metric_is_not_flagged(self):
        history = [healthy_record(step, frac_groups_pure=0.1 * step) for step in range(1, 6)]
        summary, _ = read_back(history)
        assert "frac_groups_pure" not in cast("list[str]", summary["constant_metrics"])

    def test_a_single_step_run_cannot_be_judged_constant(self):
        summary, _ = read_back([healthy_record(1)])
        assert cast("list[str]", summary["constant_metrics"]) == []

    def test_our_own_purity_metric_is_required_not_optional(self):
        """Our own purity metric is required not optional.

        The replacement has to be load-bearing, or we have swapped one unread metric for another.
        """
        assert "frac_groups_pure" in gt.REQUIRED_METRICS
        record = healthy_record(1)
        del record["frac_groups_pure"]
        _, missing = read_back([record])
        assert missing == ["frac_groups_pure"]

    def test_trl_zero_std_is_recorded_but_not_required(self):
        assert "frac_reward_zero_std" in gt.DISTRUSTED_METRICS
        assert "frac_reward_zero_std" not in gt.REQUIRED_METRICS
        record = healthy_record(1)
        del record["frac_reward_zero_std"]
        _, missing = read_back([record])
        assert missing == []

    def test_the_parse_price_guards_series_are_watched_by_the_trainer(self):
        """A guard whose per-step readings never reach the summary is a guard nobody can read.

        `games.rewards._log_parse_price_metrics` checks each failure's price against its own group at
        every step; these are the numbers that say how many failures it saw and how far below the
        group's worst parsed answer they landed. Optional rather than required because a step with no
        parse failure prices nothing.
        """
        assert set(gt.PARSE_PRICE_METRICS) <= set(gt.OPTIONAL_METRICS)
        summary, missing = read_back(
            [
                healthy_record(step, **{"parse_price/n_failures": float(step)})
                for step in range(1, 4)
            ]
        )
        assert missing == []
        assert summary["parse_price_n_failures_final"] == pytest.approx(3.0)
        assert summary["parse_price_n_failures_mean"] == pytest.approx(2.0)

    def test_the_identity_ratio_is_exempt_from_the_dead_metric_report(self):
        """Its constancy is the healthy reading, so listing it would train the reader to skip the list.

        `(worst reachable - price) / spread` is 1 at every failure by the price's arithmetic, so a run
        whose guard fired on real failures at every step reports exactly one value. The aggregates
        still land, which is what makes a drift off 1 readable after the fact -- and a genuinely dead
        metric beside it is still named.
        """
        history = [
            healthy_record(step, **{"parse_price/guard_identity_min": 1.0}) for step in range(1, 6)
        ]
        summary, _ = read_back(history)
        constant = cast("list[str]", summary["constant_metrics"])
        assert "parse_price/guard_identity_min" not in constant
        assert "frac_reward_zero_std" in constant
        assert summary["parse_price_guard_identity_min_mean"] == pytest.approx(1.0)

    def test_the_constant_mode_exempts_the_series_that_mode_pins(self):
        """Under the constant penalty the guard has nothing to check and the mean IS the knob.

        Every arm but the two prosocial-breadth ones prices failures at the constant, and the mode
        makes `n_checked`, `n_identity_checked` and `realised_mean` single-valued by construction on
        every one of them. Naming all three in the never-varied report on every run is the
        reader-desensitisation the exemption list exists to prevent, and a metric that is genuinely
        dead for some other reason is still named beside them.
        """
        history = [
            healthy_record(step, **guard_series_a_constant_mode_run_logs(step))
            for step in range(1, 6)
        ]
        summary, _ = read_back(history, parse_penalty_mode=gt.PARSE_PENALTY_CONSTANT)
        constant = cast("list[str]", summary["constant_metrics"])
        assert "parse_price/n_checked" not in constant
        assert "parse_price/n_identity_checked" not in constant
        assert "parse_price/realised_mean" not in constant
        assert "frac_reward_zero_std" in constant

    def test_the_row_relative_mode_still_names_a_guard_denominator_stuck_at_zero(self):
        """The same series under the mode where a stuck denominator is the leave-one-out hole.

        `margin-below-worse` prices from the row, so a run whose range comparison was asked of nothing
        on any step really is a guard that never ran, and a mode-blind exemption would hide exactly
        the case the split above exists for.
        """
        history = [
            healthy_record(step, **guard_series_a_constant_mode_run_logs(step))
            for step in range(1, 6)
        ]
        summary, _ = read_back(history)
        constant = cast("list[str]", summary["constant_metrics"])
        assert "parse_price/n_checked" in constant
        assert "parse_price/n_identity_checked" in constant
        assert "parse_price/realised_mean" in constant

    def test_the_identity_pair_is_exempt_under_either_mode_and_the_denominators_are_not(self):
        """The mode-independent half of the exemption, stated on the tuples themselves.

        The identity ratio is 1 by the price's arithmetic whatever the mode does, while both
        denominators are readings that a mode can pin and a defect can flatten, so only the constant
        mode may exempt them.
        """
        assert "parse_price/n_identity_checked" in gt.PARSE_PRICE_METRICS
        for mode in (gt.PARSE_PENALTY_CONSTANT, gt.PARSE_PENALTY_MARGIN_BELOW_WORSE):
            exempt = gt.constant_by_construction_metrics(mode)
            assert set(gt.PARSE_PRICE_IDENTITY_METRICS) <= set(exempt), mode
        row_relative = gt.constant_by_construction_metrics(gt.PARSE_PENALTY_MARGIN_BELOW_WORSE)
        assert "parse_price/n_checked" not in row_relative
        assert "parse_price/n_identity_checked" not in row_relative
        assert "parse_price/realised_mean" not in row_relative

    def test_a_mode_with_no_exemption_line_raises_rather_than_taking_the_narrow_one(self):
        """A third penalty mode must say what it pins, because the quiet answer is a wrong report.

        Falling through to the row-relative set would put a newly pinned series in the never-varied
        list on every run of that mode, which is how the list stops being read.
        """
        with pytest.raises(ValueError, match="parse_penalty_mode must be one of"):
            gt.constant_by_construction_metrics("margin-below-best")


class TestRunConfigArtifact:
    def test_the_payload_round_trips_through_json(self, tmp_path: Path):
        config = make_config(arm="iterated-pd-tft", output_dir=str(tmp_path / "run"))
        payload = gt.run_config_payload(
            config,
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S", "total_vram_gib": 44.4},
            derived={"prefilled_think": True, "chat_template_kwargs": {}},
        )
        path = tmp_path / gt.RUN_CONFIG_FILENAME
        path.write_text(json.dumps(payload, indent=2, default=str))
        restored = json.loads(path.read_text())
        assert restored["arm"] == "iterated-pd-tft"
        assert restored["game_id"] == "iterated-pd-tft"
        assert restored["grading"] == "iterated-return"
        assert restored["payoff_variants"] == ["temptation-2"]
        assert restored["config"]["max_completion_tokens"] == config.max_completion_tokens
        assert restored["sizing_plan"]["episodes_per_step"] == 64
        assert restored["device"]["device_name"] == "NVIDIA L40S"
        assert restored["derived"]["prefilled_think"] is True
        assert restored["git_sha"]

    def test_the_record_says_whether_the_tree_was_edited_not_only_which_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A run launched from an edited checkout must not be indistinguishable from that commit.

        `games/train.py` carried its own `git_sha()` without a dirty check, so a corpus written by
        `games.select_prompts` in the same session recorded more provenance than the arm trained
        from it did. Both now come from `games.provenance.git_provenance`.
        """
        monkeypatch.setenv("GIT_SHA", "deadbeef")
        payload = gt.run_config_payload(
            make_config(output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )
        assert payload["git_sha"] == "deadbeef"
        assert payload["git_tree_dirty"] in {True, False, None}
        assert "git_tree_dirty" in payload


class TestGrpoConfigGotchas:
    """The house gotchas, locked down: each of these has cost a run somewhere in this repo."""

    @staticmethod
    def build() -> GRPOConfig:
        config = make_config(use_liger_kernel=False, output_dir="artifacts/games/runs/test")
        return gt._build_grpo_config(config, make_plan(), dtype=torch.float32)

    def test_the_dtype_key_is_dtype_and_never_torch_dtype(self):
        """The dtype key is dtype and never torch dtype.

        TRL ignores `torch_dtype` in favour of its own `dtype`, so the model would load in float32
        while bf16=True told the trainer to autocast.
        """
        init_kwargs = self.build().model_init_kwargs
        assert isinstance(init_kwargs, dict)
        assert "dtype" in init_kwargs
        assert "torch_dtype" not in init_kwargs

    def test_no_device_map_so_a_second_gpu_can_hold_a_vllm_server(self):
        init_kwargs = self.build().model_init_kwargs
        assert isinstance(init_kwargs, dict)
        assert "device_map" not in init_kwargs

    def test_warmup_is_a_float_ratio(self):
        assert self.build().warmup_steps == pytest.approx(0.1)

    def test_the_loss_and_reward_scaling_are_the_agreed_ones(self):
        """The 2026-08-20 estimator audit's values (grpo/estimator_defaults.py has the reasoning).

        Pinned literally rather than through the constants so that changing the repo-wide default
        fails here and forces the change to be a decision, not a drive-by.
        """
        args = self.build()
        assert args.scale_rewards == "none"
        assert args.loss_type == "dr_grpo"

    def test_the_eval_loop_stays_off(self):
        """The eval loop stays off.

        TRL's eval computes the GRPO surrogate loss, which segfaults inside Liger's fused loss.
        """
        assert self.build().eval_strategy == "no"

    def test_every_intermediate_checkpoint_is_kept(self):
        args = self.build()
        assert args.save_steps == 10
        assert args.save_total_limit == 50

    def test_the_completion_trace_is_written_every_step(self):
        """The completion trace is written every step.

        TRL's completions buffer holds one generation batch, so logging less often than every step
        leaves silent holes in the per-completion trace.
        """
        args = self.build()
        assert args.log_completions is True
        assert args.logging_steps == 1

    def test_truncated_completions_keep_their_gradient(self):
        assert self.build().mask_truncated_completions is False

    def test_the_sizing_plan_reaches_trl(self):
        args = self.build()
        plan = make_plan()
        assert args.num_generations == plan.num_generations
        assert args.per_device_train_batch_size == plan.micro_batch_size
        assert args.gradient_accumulation_steps == plan.gradient_accumulation_steps

    def test_the_sampler_matches_the_selection_sweep(self):
        args = self.build()
        assert args.temperature == pytest.approx(1.0)
        assert args.top_p == pytest.approx(1.0)
        assert args.top_k == 0


class TestVllmColocateIsTheOnlyGeneration:
    """Colocate is not a switch any more: every built run generates through the engine.

    Generation at the production shape measured ~13.5 minutes per step against ~72 through
    transformers.generate, and on 2026-08-25 a stray environment export silently bought the slow
    one for a paid arm -- so the choice itself was removed (owner decision 2026-08-26). The engine
    shares one card with the trainer, so what it costs -- a third of the VRAM, and an estimator
    caveat -- still has to be as visible as what it saves.
    """

    OUTPUT_DIR: ClassVar[str] = "artifacts/games/runs/test"

    def build(self, **overrides: object) -> GRPOConfig:
        """Build the TRL arguments for a config, exactly as a run would."""
        config = make_config(use_liger_kernel=False, output_dir=self.OUTPUT_DIR, **overrides)
        return gt._build_grpo_config(config, make_plan(), dtype=torch.float32)

    def test_every_build_names_colocate_and_the_measured_fraction(self):
        """No override needed: the default arguments already carry the engine."""
        args = self.build()
        assert args.use_vllm is True
        assert args.vllm_mode == "colocate"
        assert args.vllm_gpu_memory_utilization == pytest.approx(gt.VLLM_COLOCATE_GPU_FRACTION)
        assert args.vllm_gpu_memory_utilization == pytest.approx(0.35)

    def test_sleep_offload_is_opt_in_and_stops_trainer_placement_before_engine_build(self):
        default = self.build()
        sleep_offload = self.build(colocate_sleep_offload=True)
        assert default.vllm_enable_sleep_mode is False
        assert default.place_model_on_device is None
        assert sleep_offload.vllm_enable_sleep_mode is True
        assert sleep_offload.place_model_on_device is False
        assert isinstance(sleep_offload.model_init_kwargs, dict)
        assert sleep_offload.model_init_kwargs["device_map"] is None

    def test_the_incident_export_is_refused_at_config_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A GAMES_VLLM_COLOCATE=0 in the trainer's environment invalidates any launch outright."""
        monkeypatch.setenv(generation.VLLM_COLOCATE_ENV, "0")
        with pytest.raises(RuntimeError, match="2026-08-26"):
            make_config()

    def test_the_engine_context_is_derived_from_this_run_s_own_token_budget(self):
        """Never a constant: the engine has to hold one prompt plus one full completion, no more.

        A budget shorter than a rollout makes vLLM refuse the request mid-run; a longer one makes it
        reserve KV blocks nothing will fill, on a card it is already sharing.
        """
        config = make_config(
            max_prompt_tokens=1024,
            max_completion_tokens=32768,
            # The pre-registered estimator setting at 32k, and what the startup refusal requires.
            vllm_importance_sampling_correction=False,
        )
        assert config.vllm_max_model_length == 33792
        args = gt._build_grpo_config(
            replace(config, output_dir=self.OUTPUT_DIR), make_plan(), dtype=torch.float32
        )
        assert args.vllm_max_model_length == 33792

    def test_a_longer_completion_budget_moves_the_engine_context_with_it(self):
        short = make_config(max_completion_tokens=16384)
        long = make_config(
            max_completion_tokens=32768,
            vllm_importance_sampling_correction=False,
        )
        assert long.vllm_max_model_length - short.vllm_max_model_length == 32768 - 16384

    def test_the_estimator_correction_stays_at_trls_default(self):
        """Dropping it changes the gradient estimator, which is never a default.

        TRL's default is on; this asserts we adopt whatever that is rather than pinning a copy, so
        a version bump that changed it would be visible in the run record instead of here.
        """
        assert make_config().vllm_importance_sampling_correction is True
        assert (
            self.build().vllm_importance_sampling_correction
            is GRPOConfig(
                output_dir=self.OUTPUT_DIR, bf16=False
            ).vllm_importance_sampling_correction
        )

    def test_dropping_the_estimator_correction_reaches_trl(self):
        args = self.build(vllm_importance_sampling_correction=False)
        assert args.vllm_importance_sampling_correction is False

    def test_the_correction_mode_stays_at_trls_default_and_reaches_trl_when_moved(self):
        """The mode decides what the correction DOES with the log-probability difference, so it is
        passed rather than left implicit: a run whose record names token_truncate while TRL kept its
        sequence-level default would report a treatment it never ran."""
        default = GRPOConfig(output_dir=self.OUTPUT_DIR, bf16=False).vllm_importance_sampling_mode
        assert self.build().vllm_importance_sampling_mode == default
        assert (
            self.build(vllm_importance_sampling_mode="token_truncate").vllm_importance_sampling_mode
            == "token_truncate"
        )

    def test_the_sampler_record_reads_the_colocate_mode_off_these_arguments(self):
        """The seam: GRPOTrainer copies `use_vllm` and `vllm_mode` straight off its arguments.

        `log_resolved_sampler` reports the backend from those two trainer attributes, so a run's
        recorded `generation_backend` is only as truthful as what this builder puts in them.
        """
        args = self.build()
        trainer = sampler_trainer(use_vllm=args.use_vllm, vllm_mode=args.vllm_mode)
        del trainer.generation_config
        assert gt.log_resolved_sampler(trainer)["generation_backend"] == "vllm-colocate"  # pyright: ignore[reportArgumentType]

    def test_the_sampler_record_still_names_a_foreign_transformers_trainer_honestly(self):
        """`log_resolved_sampler` records whatever it is handed -- diagnosis, not enforcement.

        The reward-hacking trainer shares this recorder and keeps a deliberate transformers path;
        enforcement for games is `assert_trainer_generates_through_vllm`, tested just below.
        """
        sampler = gt.log_resolved_sampler(sampler_trainer(use_vllm=False))  # pyright: ignore[reportArgumentType]
        assert sampler["generation_backend"] == "transformers-generate"

    def test_a_built_trainer_on_the_transformers_path_is_refused_by_the_consumption_check(self):
        """The last line of the guard reads the built trainer, not the env or the config."""
        trainer = sampler_trainer(use_vllm=False)
        with pytest.raises(RuntimeError, match="transformers-generate"):
            preflight.assert_trainer_generates_through_vllm(trainer)  # pyright: ignore[reportArgumentType]

    def test_a_built_trainer_on_a_non_colocate_engine_is_refused_too(self):
        trainer = sampler_trainer(use_vllm=True, vllm_mode="server")
        with pytest.raises(RuntimeError, match="vllm-server"):
            preflight.assert_trainer_generates_through_vllm(trainer)  # pyright: ignore[reportArgumentType]

    def test_a_colocate_trainer_passes_the_consumption_check(self):
        trainer = sampler_trainer(use_vllm=True, vllm_mode="colocate")
        preflight.assert_trainer_generates_through_vllm(trainer)  # pyright: ignore[reportArgumentType]

    def test_a_run_without_vllm_refuses_before_a_card_is_reserved(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """TRL raises too, but only after the weights load -- minutes of billing later.

        With the backend choice gone this fires on a plain default config: a box without the
        engine cannot train at all, and the message says to install it rather than fall back.
        """
        monkeypatch.setattr(gt.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(ValueError, match="vllm is not installed"):
            make_config()

    @pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5, -0.1])
    def test_a_fraction_that_leaves_the_trainer_nothing_is_refused(self, fraction: float):
        with pytest.raises(ValueError, match="vllm_gpu_memory_utilization"):
            make_config(vllm_gpu_memory_utilization=fraction)


class TestTheCorrectionRefusesTheShapeThatOoms:
    """Correction-ON colocate dies mid-run when the old-logps pass is wider than the card can hold.

    Measured 2026-08-19, twice, on a 95 GiB card: TRL's own unchunked pass wanted 30.31 GiB of fp32
    logits for ONE 32,768-token row beside the resident engine, and did not fit even with all three
    TRL memory patches applied. That measurement set the 16 GiB threshold, and the threshold has not
    moved -- what moved is the shape being priced against it. `InstrumentedGRPOTrainer` computes the
    pass over `old_logps_chunk_tokens`-position slices, so the widest logits tensor a run allocates
    is one chunk rather than one row.

    What the refusal prices is not that widest tensor but how many of them are LIVE at the peak:
    `old_logps_pass_peak_gib` counts one divided fp32 copy per micro-batch row plus the logsumexp
    temporary inside `selective_log_softmax`, so a single-row micro-batch pays twice the per-slice
    figure. A chunk at or above the row length IS the unchunked shape and is still refused; so is the
    16,384-position chunk an earlier per-slice pricing allowed at "15.16 GiB", which really prices at
    30.31 -- the dead shape's own number. The default 2,048-token chunk is 3.79 GiB at one row and is
    accepted at every budget.

    Sabotage run once and watched red (2026-09-04): pricing one copy per slice again, the arithmetic
    this class corrects, failed eight tests across this class and the banner's.
    """

    @pytest.fixture(autouse=True)
    def engine_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin vllm as installed, so these assert the correction refusal and not the import one."""
        monkeypatch.setattr(gt.importlib.util, "find_spec", lambda name: object())

    def test_the_unchunked_shape_that_died_is_still_refused_at_startup(self):
        with pytest.raises(ValueError, match="importance-sampling correction"):
            make_config(max_completion_tokens=32768, old_logps_chunk_tokens=32768)

    def test_a_chunk_above_the_row_length_is_refused_too(self):
        """Clamped to the row, so it is the unchunked shape by another spelling."""
        with pytest.raises(ValueError, match="importance-sampling correction"):
            make_config(max_completion_tokens=32768, old_logps_chunk_tokens=65536)

    def test_the_refusal_names_both_escape_hatches(self):
        with pytest.raises(ValueError, match="--no-vllm-importance-sampling-correction"):
            make_config(max_completion_tokens=32768, old_logps_chunk_tokens=32768)
        with pytest.raises(ValueError, match=gt.VLLM_IS_CORRECTION_ENV):
            make_config(max_completion_tokens=32768, old_logps_chunk_tokens=32768)

    def test_the_refusal_names_the_chunk_it_priced(self):
        with pytest.raises(ValueError, match="--old-logps-chunk-tokens"):
            make_config(max_completion_tokens=32768, old_logps_chunk_tokens=32768)

    def test_the_default_chunk_is_accepted_at_the_budget_that_died(self):
        """The whole point of the chunked pass: correction ON at 32k is now a runnable shape."""
        config = make_config(max_completion_tokens=32768)
        assert config.vllm_importance_sampling_correction is True
        assert config.old_logps_chunk_tokens == gt.OLD_LOGPS_CHUNK_TOKENS

    def test_log_only_is_accepted_at_the_budget_that_died(self):
        """The five-step probe's own shape: the correction computed and logged, the weight left at 1."""
        config = make_config(max_completion_tokens=32768, vllm_importance_sampling_log_only=True)
        assert config.vllm_importance_sampling_correction is True

    def test_correction_off_passes_the_same_shape(self):
        config = make_config(
            max_completion_tokens=32768,
            vllm_importance_sampling_correction=False,
        )
        assert config.vllm_max_model_length == 1024 + 32768

    def test_a_smoke_sized_budget_keeps_trls_default_runnable(self):
        """~1.9 GiB of row logits at 2,048 tokens ran fine; the refusal must not reach it."""
        config = make_config(max_completion_tokens=2048, allow_short_completions=True)
        assert config.vllm_importance_sampling_correction is True

    def test_a_chunk_wider_than_the_row_is_priced_at_the_row(self):
        """`chunked_per_token_logps` slices `range(0, positions, chunk)`, so one slice is the row.

        The clamp is load-bearing rather than tidiness: priced at the flag's own value this shape
        reads as 60.62 GiB and would be refused, while the pass can never build more than the 2,048
        positions the row has.
        """
        config = make_config(
            max_completion_tokens=2048,
            allow_short_completions=True,
            old_logps_chunk_tokens=32768,
        )
        assert config.vllm_importance_sampling_correction is True

    def test_the_boundary_arithmetic_is_the_documented_formula(self):
        """Sabotage guard for the threshold: the default chunk is under it and the 4B floor is over.

        Priced per LIVE copy rather than per slice, which is the correction this class's docstring
        describes: the 16,384-position chunk reads as 15.16 GiB per slice and 30.31 GiB at the peak,
        and 30.31 is the figure the pass was measured dead at.
        """
        default_gib = gt.old_logps_pass_peak_gib(chunk_tokens=gt.OLD_LOGPS_CHUNK_TOKENS, rows=1)
        floor_gib = gt.old_logps_pass_peak_gib(chunk_tokens=16384, rows=1)
        dead_gib = gt.old_logps_pass_peak_gib(chunk_tokens=32768, rows=1)
        assert default_gib < gt.IS_CORRECTION_LOGITS_REFUSAL_GIB < floor_gib < dead_gib
        assert default_gib == pytest.approx(3.79, abs=0.01)
        assert floor_gib == pytest.approx(30.31, abs=0.05)
        assert dead_gib == pytest.approx(60.62, abs=0.05)

    def test_the_peak_counts_one_copy_per_row_plus_the_reduction_temporary(self):
        """The arithmetic in one place: `rows + 1` chunk-sized fp32 copies, never one."""
        one_copy = 2048 * gt.QWEN3_5_VOCAB_SIZE * gt.BYTES_PER_FLOAT32 / gt.BYTES_PER_GIB
        for rows in (1, 2, 4, 8):
            assert gt.old_logps_pass_peak_gib(chunk_tokens=2048, rows=rows) == pytest.approx(
                (rows + 1) * one_copy
            )
        assert one_copy == pytest.approx(1.89, abs=0.01)

    def test_the_4b_floor_budget_chunk_is_refused_rather_than_allowed(self):
        """It priced at 15.16 GiB per slice and passed; at the peak it is the shape measured dead."""
        with pytest.raises(ValueError, match=r"30\.3 GiB"):
            make_config(max_completion_tokens=32768, old_logps_chunk_tokens=16384)

    def test_a_micro_batch_the_launch_named_multiplies_the_price(self):
        """Seven rows of the default chunk still fit; eight do not, and TRL runs the pass per row."""
        config = make_config(max_completion_tokens=32768, micro_batch_size=7)
        assert config.micro_batch_size == 7
        with pytest.raises(ValueError, match="9 copies"):
            make_config(max_completion_tokens=32768, micro_batch_size=8)

    def test_the_derived_micro_batch_is_repriced_once_the_plan_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The hole an unpriced row factor leaves: `micro_batch_size` unset, so the card decides it.

        `_derive_plan` re-prices against the number `plan_sizing` derived, which is what TRL passes as
        the pass's `batch_size`, and it does so before any weights load.
        """
        monkeypatch.setattr(
            gt,
            "checkpoint_sequence_cost",
            lambda _model: sizing.sequence_cost(hybrid_text_config()),
        )
        monkeypatch.setattr(
            gt, "plan_sizing", lambda **_kwargs: replace(make_plan(), micro_batch_size=8)
        )
        config = make_config(max_completion_tokens=32768)
        assert config.micro_batch_size is None
        with pytest.raises(ValueError, match="the micro-batch the sizing plan derived"):
            gt._derive_plan(  # pyright: ignore[reportPrivateUsage]
                config,
                device={
                    "total_vram_gib": 95.0,
                    "free_vram_gib_at_start": 60.0,
                    "device_name": "NVIDIA H100",
                },
                dtype=torch.bfloat16,
                param_count=9_000_000_000,
            )

    def test_a_plan_within_the_line_is_announced_and_allowed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """The positive control on the re-pricing, and on the banner that reports the same number."""
        monkeypatch.setattr(
            gt,
            "checkpoint_sequence_cost",
            lambda _model: sizing.sequence_cost(hybrid_text_config()),
        )
        monkeypatch.setattr(
            gt, "plan_sizing", lambda **_kwargs: replace(make_plan(), micro_batch_size=1)
        )
        with caplog.at_level(logging.INFO, logger="games.train"):
            plan, _cost = gt._derive_plan(  # pyright: ignore[reportPrivateUsage]
                make_config(max_completion_tokens=32768),
                device={
                    "total_vram_gib": 95.0,
                    "free_vram_gib_at_start": 60.0,
                    "device_name": "NVIDIA H100",
                },
                dtype=torch.bfloat16,
                param_count=9_000_000_000,
            )
        assert plan.micro_batch_size == 1
        assert "3.79 GiB" in caplog.text


class TestColocateSizesTheCardItShares:
    """A plan built on the whole card would budget episodes into the engine's own memory.

    Sizing runs before `GRPOTrainer` exists and therefore before the engine does, so the free-VRAM
    reading it is given describes a card about to lose a third of itself.
    """

    def test_the_engine_takes_a_fraction_of_total_not_of_free(self):
        """vLLM's own convention, and the reason this is not computed off the free-VRAM reading."""
        assert sizing.colocate_reserved_gib(
            total_vram_gib=95.0, gpu_memory_utilization=0.35
        ) == pytest.approx(33.25)

    def test_a_colocate_plan_is_smaller_than_the_same_plan_on_the_whole_card(self):
        whole = make_plan(free_vram_gib=44.0, weights_gib=8.0)
        shared = make_plan(free_vram_gib=44.0, weights_gib=8.0, engine_reserved_gib=15.0)
        assert shared.usable_vram_gib < whole.usable_vram_gib
        assert shared.episodes_that_fit < whole.episodes_that_fit
        assert shared.engine_reserved_gib == pytest.approx(15.0)


class TestASelfSchedulingEngineIsNotChargedTwice:
    """Under colocate the episodes' memory lives in the engine's reservation, already subtracted.

    Charging the HF per-episode arithmetic on top of `engine_reserved_gib` double-counted the same
    memory: the reservation pays for the episodes, and the per-episode terms (KV cache, DeltaNet
    state, prefill upcast) price a transformers.generate path a colocate run never takes. The
    trainer's own ceiling is the micro-batch token budget, which is derived either way.
    """

    TIGHT_CARD: ClassVar[dict[str, float]] = {"free_vram_gib": 16.0, "weights_gib": 8.0}

    def test_the_hf_path_still_clamps_on_the_same_shape(self):
        """The control: without the flag, this exact shape is clamped, so the flag is what acts."""
        plan = make_plan(**self.TIGHT_CARD)
        assert plan.clamped
        assert plan.episodes_per_step < 64

    def test_the_episode_clamp_is_skipped_when_the_engine_schedules(self):
        plan = make_plan(**self.TIGHT_CARD, generation_schedules_own_batch=True)
        assert not plan.clamped
        assert plan.episodes_per_step == 64
        assert plan.num_generations == 8
        assert plan.prompts_per_step == 8

    def test_the_arithmetic_is_still_reported_not_erased(self):
        """`episodes_that_fit` keeps the HF answer on the record; only the clamp is skipped."""
        plan = make_plan(**self.TIGHT_CARD, generation_schedules_own_batch=True)
        assert plan.episodes_that_fit < 64
        assert "paged cache" in plan.reason

    def test_a_shape_that_fits_anyway_reads_exactly_as_before(self):
        roomy = make_plan(free_vram_gib=44.0, weights_gib=8.0)
        engine = make_plan(free_vram_gib=44.0, weights_gib=8.0, generation_schedules_own_batch=True)
        assert engine.episodes_per_step == roomy.episodes_per_step
        assert engine.reason == roomy.reason

    def test_the_engine_reservation_is_still_subtracted_first(self):
        """Skipping the clamp must not resurrect budgeting into the engine's own block."""
        with pytest.raises(RuntimeError, match="resident engine"):
            make_plan(
                free_vram_gib=16.0,
                weights_gib=8.0,
                engine_reserved_gib=17.0,
                generation_schedules_own_batch=True,
            )

    def test_the_headroom_is_the_card_less_the_engine(self):
        shared = make_plan(free_vram_gib=44.0, weights_gib=8.0, engine_reserved_gib=15.0)
        assert shared.episode_headroom_gib == pytest.approx(
            (44.0 - 15.0) * 0.9 - 8.0 - sizing.UNMODELLED_OVERHEAD_GIB
        )

    def test_the_micro_batch_budget_also_gives_up_the_engine_s_share(self):
        """Generation and the training pass fail differently, so both budgets have to shrink.

        The episode count fails in prefill and the training pass fails inside a LoRA projection; a
        plan that corrected only the first still OOMs on a card it is sharing.
        """
        whole = sizing.micro_batch_token_budget(free_vram_gib=44.0, weights_gib=8.0)
        shared = make_plan(free_vram_gib=44.0, weights_gib=8.0, engine_reserved_gib=15.0)
        assert f"{whole}-token" not in shared.reason

    def test_the_plan_says_the_engine_took_its_share(self):
        shared = make_plan(free_vram_gib=44.0, weights_gib=8.0, engine_reserved_gib=15.0)
        assert "resident engine" in shared.reason
        assert make_plan(free_vram_gib=44.0, weights_gib=8.0).engine_reserved_gib == 0.0

    def test_an_engine_holding_the_whole_card_is_refused_by_name(self):
        with pytest.raises(RuntimeError, match="leaves nothing"):
            make_plan(free_vram_gib=20.0, weights_gib=8.0, engine_reserved_gib=20.0)

    def test_the_default_path_is_unchanged_by_the_new_parameter(self):
        explicit_zero = make_plan(free_vram_gib=44.0, weights_gib=8.0, engine_reserved_gib=0.0)
        assert explicit_zero == make_plan(free_vram_gib=44.0, weights_gib=8.0)


class TestTheColocateEnvironmentIsTheSingleSourceOfTruth:
    """The engine knobs are read from one place, and the backend is not a knob at all."""

    def test_a_confirming_environment_parses_clean(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(generation.VLLM_COLOCATE_ENV, "1")
        assert gt._parse_args(["--arm", "dictator", "--generate-fresh"]).arm == "dictator"

    def test_the_incident_export_is_refused_at_parse_time(self, monkeypatch: pytest.MonkeyPatch):
        """The exact environment that bought the 2026-08-25 slow arm cannot parse a config."""
        monkeypatch.setenv(generation.VLLM_COLOCATE_ENV, "0")
        with pytest.raises(RuntimeError, match="2026-08-26"):
            gt._parse_args(["--arm", "dictator", "--generate-fresh"])

    @pytest.mark.parametrize("flag", ["--vllm-colocate", "--no-vllm-colocate"])
    def test_the_removed_backend_flags_do_not_parse(self, flag: str):
        """argparse exits on the retired switch, so a stale launch script fails at startup."""
        with pytest.raises(SystemExit):
            gt._parse_args(["--arm", "dictator", "--generate-fresh", flag])

    def test_the_fraction_and_the_estimator_come_from_the_environment_too(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(gt.VLLM_GPU_FRACTION_ENV, "0.4")
        monkeypatch.setenv(gt.VLLM_IS_CORRECTION_ENV, "0")
        config = gt._parse_args(["--arm", "dictator", "--generate-fresh"])
        assert config.vllm_gpu_memory_utilization == pytest.approx(0.4)
        assert config.vllm_importance_sampling_correction is False

    def test_a_flag_overrides_the_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(gt.VLLM_GPU_FRACTION_ENV, "0.4")
        config = gt._parse_args(
            [
                "--arm",
                "dictator",
                "--generate-fresh",
                "--vllm-gpu-memory-utilization",
                "0.5",
            ]
        )
        assert config.vllm_gpu_memory_utilization == pytest.approx(0.5)


def announce_colocate(config: gt.GameTrainConfig, *, rows: int = 1) -> None:
    """Call the banner exactly as `_derive_plan` does, with the instrument values passed in.

    The banner takes those values rather than reading them off the config, because
    `reward_hacking.train` hands it a config that has neither field; the helper keeps every test here
    calling it the way production does instead of restating the argument list per test.
    """
    chunk_tokens = min(config.max_completion_tokens, config.old_logps_chunk_tokens)
    gt.log_colocate_settings(
        config,
        engine_reserved_gib=33.2,
        old_logps_chunk_tokens=chunk_tokens,
        old_logps_peak_gib=gt.old_logps_pass_peak_gib(chunk_tokens=chunk_tokens, rows=rows),
        importance_sampling_log_only=config.vllm_importance_sampling_log_only,
        importance_sampling_mode=config.vllm_importance_sampling_mode,
    )


class TestTheColocateBannerSaysWhatWasTraded:
    """Both estimator settings have a price at the production shape, so both are announced."""

    def test_dropping_the_correction_states_the_caveat_it_carries(
        self, caplog: pytest.LogCaptureFixture
    ):
        config = make_config(vllm_importance_sampling_correction=False)
        with caplog.at_level(logging.INFO, logger="games.train"):
            announce_colocate(config)
        assert "uncorrected" in caplog.text
        assert "not comparable" in caplog.text
        assert "unaffected" in caplog.text

    def test_keeping_the_correction_warns_about_the_pass_that_ran_out_of_memory(
        self, caplog: pytest.LogCaptureFixture
    ):
        config = make_config()
        with caplog.at_level(logging.INFO, logger="games.train"):
            announce_colocate(config)
        assert "30.31 GiB" in caplog.text
        assert gt.VLLM_IS_CORRECTION_ENV in caplog.text

    def test_the_engine_s_share_and_context_are_both_in_the_log(
        self, caplog: pytest.LogCaptureFixture
    ):
        config = make_config(
            max_completion_tokens=32768,
            # The correction-OFF branch of the banner, at the budget whose unchunked pass died.
            vllm_importance_sampling_correction=False,
        )
        with caplog.at_level(logging.INFO, logger="games.train"):
            announce_colocate(config)
        assert "33.2" in caplog.text
        assert "33792" in caplog.text

    def test_every_run_announces_the_engine_it_generates_through(
        self, caplog: pytest.LogCaptureFixture
    ):
        """No silent branch left: a default config's banner still names the colocated engine."""
        with caplog.at_level(logging.INFO, logger="games.train"):
            announce_colocate(make_config())
        assert "colocated vLLM engine" in caplog.text

    def test_the_announced_figure_is_the_live_peak_not_one_slice(
        self, caplog: pytest.LogCaptureFixture
    ):
        """The banner used to print one slice's cost, which is about half of what the pass holds."""
        with caplog.at_level(logging.INFO, logger="games.train"):
            announce_colocate(make_config())
        assert "3.79 GiB" in caplog.text
        assert "1.90 GiB" not in caplog.text

    def test_a_wider_micro_batch_is_announced_as_a_bigger_peak(
        self, caplog: pytest.LogCaptureFixture
    ):
        """One divided fp32 copy per row, so the figure has to move with the rows in flight."""
        with caplog.at_level(logging.INFO, logger="games.train"):
            announce_colocate(make_config(), rows=4)
        assert "9.47 GiB" in caplog.text

    def test_the_correction_mode_is_named_so_a_box_log_says_which_estimator_ran(
        self, caplog: pytest.LogCaptureFixture
    ):
        """The mode is the difference between clipping a token's ratio and discarding a rollout, and
        the box log is the copy that survives a run killed before it writes a summary."""
        with caplog.at_level(logging.INFO, logger="games.train"):
            announce_colocate(make_config(vllm_importance_sampling_mode="token_truncate"))
        assert "token_truncate" in caplog.text

    def test_the_default_mode_is_named_too_so_the_two_are_diffable(
        self, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.INFO, logger="games.train"):
            announce_colocate(make_config())
        assert VLLM_IMPORTANCE_SAMPLING_MODE in caplog.text

    def test_log_only_is_announced_as_a_measurement_rather_than_a_treatment(
        self, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.INFO, logger="games.train"):
            announce_colocate(make_config(vllm_importance_sampling_log_only=True))
        assert "log-only importance sampling" in caplog.text
        assert "correction-OFF" in caplog.text

    def test_the_banner_reads_nothing_off_the_config_but_the_engine_contract(
        self, caplog: pytest.LogCaptureFixture
    ):
        """The regression this contract exists for: `reward_hacking.train` has neither instrument
        field, so the banner must run against an object carrying only the three it declares."""
        engine_only = SimpleNamespace(
            vllm_gpu_memory_utilization=0.33,
            vllm_max_model_length=33792,
            vllm_importance_sampling_correction=True,
        )
        with caplog.at_level(logging.INFO, logger="games.train"):
            gt.log_colocate_settings(
                cast("gt.ColocateEngineSettings", engine_only),
                engine_reserved_gib=33.2,
                old_logps_chunk_tokens=gt.OLD_LOGPS_CHUNK_TOKENS,
                old_logps_peak_gib=3.79,
                importance_sampling_log_only=False,
            )
        assert "correction is ON" in caplog.text
        assert "3.79 GiB" in caplog.text


def sampler_trainer(**overrides: Any) -> SimpleNamespace:
    """A constructed-trainer stand-in carrying only what the sampler record reads.

    Every attribute here is one `GRPOTrainer.__init__` sets. `generation_config` is the exception
    that matters: TRL builds it on the transformers path only, so a vLLM stub omits it entirely
    rather than setting it to None.
    """
    base: dict[str, Any] = {
        "args": SimpleNamespace(generation_kwargs=None),
        "use_vllm": False,
        "vllm_mode": "colocate",
        "num_generations": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": None,
        "repetition_penalty": 1.0,
        "max_completion_length": 32768,
        "generation_config": SimpleNamespace(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=None,
            repetition_penalty=1.0,
            max_new_tokens=32768,
            do_sample=True,
            cache_implementation=None,
        ),
    }
    return SimpleNamespace(**(base | overrides))


class TestResolvedSamplerRecord:
    """What the two sides of a rollout comparison sampled with, read off the built trainer.

    Reconstructing this from a tarball of the code a box happened to be running is what an
    HF-versus-vLLM comparison had to do once, and the reason it was possible at all was luck.
    """

    def test_the_transformers_path_reports_the_generation_config_not_our_own_inputs(self):
        """The drift case, and the reason the record is read off the trainer at all.

        TRL folds `generation_kwargs` into the GenerationConfig, and that object rather than the
        trainer's own attributes is what generate() reads. A record trusting `trainer.temperature`
        would report 1.0 for a run that actually sampled at 0.7.
        """
        trainer = sampler_trainer(
            args=SimpleNamespace(generation_kwargs={"temperature": 0.7}),
            generation_config=SimpleNamespace(
                temperature=0.7,
                top_p=0.95,
                top_k=0,
                min_p=None,
                repetition_penalty=1.05,
                max_new_tokens=4096,
                do_sample=True,
                cache_implementation="static",
            ),
        )
        sampler = gt.log_resolved_sampler(trainer)  # pyright: ignore[reportArgumentType]
        assert sampler["temperature"] == pytest.approx(0.7)
        assert sampler["top_p"] == pytest.approx(0.95)
        assert sampler["repetition_penalty"] == pytest.approx(1.05)
        assert sampler["max_new_tokens"] == 4096
        assert sampler["do_sample"] is True
        assert sampler["cache_implementation"] == "static"
        assert sampler["generation_kwargs"] == {"temperature": 0.7}
        assert sampler["generation_backend"] == "transformers-generate"

    def test_the_vllm_path_names_its_mode_and_has_no_generation_config_to_read(self):
        trainer = sampler_trainer(use_vllm=True, vllm_mode="colocate", temperature=0.9)
        del trainer.generation_config
        sampler = gt.log_resolved_sampler(trainer)  # pyright: ignore[reportArgumentType]
        assert sampler["generation_backend"] == "vllm-colocate"
        assert sampler["temperature"] == pytest.approx(0.9)
        assert sampler["do_sample"] is None
        assert sampler["cache_implementation"] is None

    def test_the_server_mode_is_distinguishable_from_colocate(self):
        trainer = sampler_trainer(use_vllm=True, vllm_mode="server")
        del trainer.generation_config
        assert (
            gt.log_resolved_sampler(trainer)["generation_backend"]  # pyright: ignore[reportArgumentType]
            == "vllm-server"
        )

    def test_both_backends_record_the_same_keys_so_two_arms_diff(self):
        vllm = sampler_trainer(use_vllm=True)
        del vllm.generation_config
        transformers_keys = set(gt.log_resolved_sampler(sampler_trainer()))  # pyright: ignore[reportArgumentType]
        assert transformers_keys == set(gt.log_resolved_sampler(vllm))  # pyright: ignore[reportArgumentType]

    def test_trl_defaults_nobody_passes_are_recorded_too(self):
        """`min_p` and `repetition_penalty` are whatever the installed TRL defaults to.

        Nothing in this repo sets either, which is exactly why the record has to carry them: a
        version bump moving one would otherwise change the rollouts with nothing in the log.
        """
        args = TestGrpoConfigGotchas.build()
        trainer = sampler_trainer(min_p=args.min_p, repetition_penalty=args.repetition_penalty)
        sampler = gt.log_resolved_sampler(trainer)  # pyright: ignore[reportArgumentType]
        assert "min_p" in sampler
        assert sampler["repetition_penalty"] == pytest.approx(args.repetition_penalty)

    def test_one_info_line_carries_the_whole_record(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.INFO, logger="games.preflight"):
            sampler = gt.log_resolved_sampler(sampler_trainer())  # pyright: ignore[reportArgumentType]
        lines = [line for line in caplog.text.splitlines() if "resolved sampler" in line]
        assert len(lines) == 1
        for key in sampler:
            assert key in lines[0]


class TestStatedMatchMixtureGuard:
    """The vs-stated-match arm's startup gates: the probability column and the two-sided mixture.

    Both refusals exist because the failure they catch produces a full set of plausible artifacts:
    `games.rewards` does raise on the unset marker, but only inside the reward function after a
    generation batch is paid for, and a one-sided corpus never raises anywhere -- it just trains a
    fixed function of the model's own action, the exact counterpart-blindness the arm exists to
    avoid, behind healthy-looking curves.
    """

    @staticmethod
    def row(**overrides: Any) -> dict[str, Any]:
        base = {
            "prompt": "a sheet of paper",
            "prompt_id": "pd-track-record-0001",
            "game_id": "pd-track-record",
            "grading": "vs-stated-match",
            # Normalised twin-PD cells: crossover at 5/7.
            "payoff_cc": 0.6,
            "payoff_cd": 0.0,
            "payoff_dc": 1.0,
            "payoff_dd": 0.2,
            "stated_match_prob": 0.95,
        }
        return base | overrides

    def test_a_two_sided_corpus_passes(self) -> None:
        rows = [
            self.row(prompt_id="high", stated_match_prob=0.95),
            self.row(prompt_id="low", stated_match_prob=0.4),
        ]
        gt.assert_stated_match_mixture(rows, gt.ARMS["pd-track-record"])

    def test_the_unset_marker_is_refused_before_any_weights_load(self) -> None:
        rows = [self.row(), self.row(prompt_id="bad", stated_match_prob=-1.0)]
        with pytest.raises(ValueError, match="stated_match_prob"):
            gt.assert_stated_match_mixture(rows, gt.ARMS["pd-track-record"])

    def test_a_missing_column_is_refused_the_same_way(self) -> None:
        incomplete = self.row()
        del incomplete["stated_match_prob"]
        with pytest.raises(ValueError, match="stated_match_prob"):
            gt.assert_stated_match_mixture([incomplete], gt.ARMS["pd-track-record"])

    def test_a_one_sided_corpus_is_refused_as_the_fixed_p_design(self) -> None:
        """A one sided corpus is refused as the fixed p design.

        Every row above the crossover: cooperation is unconditionally optimal, so the stated track
        record never changes the answer and the mixture is a label, not a mechanism.
        """
        rows = [
            self.row(prompt_id="p95", stated_match_prob=0.95),
            self.row(prompt_id="p99", stated_match_prob=0.99),
        ]
        with pytest.raises(ValueError, match="same action EV-optimal"):
            gt.assert_stated_match_mixture(rows, gt.ARMS["pd-track-record"])

    def test_a_row_exactly_on_its_crossover_is_refused(self) -> None:
        """A row exactly on its crossover is refused.

        Dyadic cells on purpose: the gap at the crossover is only EXACTLY zero when every operand
        is a power-of-two fraction (twin-pd's 5/7 leaves a one-ulp residue, which is why the real
        corpus can never land here and why the audit floor is a margin, not an equality test).
        Cells 0.75/0.25/0.75/0.25 put the crossover at exactly 0.5.
        """
        knife_edge = self.row(
            prompt_id="knife-edge",
            payoff_cc=0.75,
            payoff_cd=0.25,
            payoff_dc=0.75,
            payoff_dd=0.25,
            stated_match_prob=0.5,
        )
        rows = [self.row(prompt_id="low", stated_match_prob=0.4), knife_edge]
        with pytest.raises(ValueError, match="exactly on its EV crossover"):
            gt.assert_stated_match_mixture(rows, gt.ARMS["pd-track-record"])

    def test_other_gradings_are_untouched(self) -> None:
        """Other gradings are untouched.

        The guard is the stated-match arm's own; a group-mix corpus carries the marker on every row
        and must pass through unexamined.
        """
        rows = [self.row(grading="group-mix", stated_match_prob=-1.0)]
        gt.assert_stated_match_mixture(rows, gt.ARMS["twin-pd-group"])


class TestStatedMatchBackfillOnLoad:
    def test_an_old_corpus_gets_the_unset_marker_backfilled(self, tmp_path: Path) -> None:
        """Corpora written before the column existed stay trainable, with the marker filled in.

        The marker is what every current row builder writes for a prompt stating no track
        record, and the only reader of the column refuses it -- so the backfill cannot mis-grade
        anything, it only keeps the reward function's column requirement satisfiable.
        """
        row = {
            "prompt": "a sheet of paper",
            "prompt_id": "twin-pd-0001",
            "game_id": "twin-pd",
            "grading": "group-mix",
        }
        path = tmp_path / "old-corpus.jsonl"
        path.write_text(json.dumps(row) + "\n")
        loaded = gt.load_corpus(str(path), gt.ARMS["twin-pd-group"])
        assert loaded[0]["stated_match_prob"] == gt.STATED_MATCH_PROB_UNSET

    def test_a_corpus_carrying_the_column_is_left_alone(self, tmp_path: Path) -> None:
        row = {
            "prompt": "a sheet of paper",
            "prompt_id": "twin-pd-0001",
            "game_id": "twin-pd",
            "grading": "group-mix",
            "stated_match_prob": 0.25,
        }
        path = tmp_path / "corpus.jsonl"
        path.write_text(json.dumps(row) + "\n")
        loaded = gt.load_corpus(str(path), gt.ARMS["twin-pd-group"])
        assert loaded[0]["stated_match_prob"] == 0.25


class TestParsePenaltyModeIsTheArms:
    """How an unparseable completion is priced is part of the grading, so the arm carries it.

    Two arms over one corpus that differ only in this are two experiments (the registry's
    duplicate-key test says so), and `arm` is already a resume-identity field, so a checkpoint
    priced one way cannot continue under the other. The run record names the mode at the top
    level beside the grading, where the box's config watcher and the eval-trace meta read it.
    """

    def test_the_run_record_names_the_arms_mode_beside_the_grading(self, tmp_path: Path):
        softpen = make_config(
            arm="pd-track-record-v2-softpen",
            generate_fresh=False,
            corpus_path="c.jsonl",
            output_dir=str(tmp_path / "run"),
        )
        payload = gt.run_config_payload(
            softpen, plan=make_plan(), device={"device_name": "NVIDIA L40S"}, derived={}
        )
        assert payload["grading"] == "vs-stated-match"
        assert payload["parse_penalty_mode"] == "margin-below-worse"
        config_record = payload["config"]
        assert isinstance(config_record, dict)
        assert "parse_penalty_mode" not in config_record

    def test_every_earlier_arm_records_the_constant(self, tmp_path: Path):
        payload = gt.run_config_payload(
            make_config(output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )
        assert payload["parse_penalty_mode"] == "constant"


class TestTheOptimizerEpsilonAndAdapterDropoutAreRecordedTreatment:
    """Two knobs the wave-4b arm moves, and moves as explicit recorded flags rather than defaults.

    The GRPO best-practices note measured the banked 9B self arm's step-70 optimizer state and found
    99.3 percent of the LoRA A-matrix second moments below AdamW's default epsilon of 1e-8, damping
    the A side of every adapter to about a tenth of its nominal step; and adapter dropout makes the
    graded forward stochastic while vLLM sampled the rollout without it. Both are therefore
    treatment changes against the banked pair, so both stay at the banked values by default, land in
    `run_config.json` and join the resume identity: a checkpoint's steps trained under one optimizer
    epsilon or one adapter dropout cannot be continued under another without saying so.
    """

    def test_the_defaults_are_the_values_the_banked_pair_trained_under(self):
        config = make_config()
        assert config.adam_epsilon == pytest.approx(1e-8)
        assert config.lora_dropout == pytest.approx(0.05)

    def test_both_flags_parse_onto_their_config_fields(self):
        config = gt._parse_args(
            [
                "--arm",
                "twin-pd-group",
                "--generate-fresh",
                "--adam-epsilon",
                "1e-15",
                "--lora-dropout",
                "0",
            ]
        )
        assert config.adam_epsilon == pytest.approx(1e-15)
        assert config.lora_dropout == pytest.approx(0.0)

    def test_both_land_in_the_run_record_where_the_config_watcher_reads_them(self, tmp_path: Path):
        payload = gt.run_config_payload(
            make_config(adam_epsilon=1e-15, lora_dropout=0.0, output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )
        recorded = payload["config"]
        assert isinstance(recorded, dict)
        assert recorded["adam_epsilon"] == pytest.approx(1e-15)
        assert recorded["lora_dropout"] == pytest.approx(0.0)

    def test_the_epsilon_reaches_trl_rather_than_only_the_record(self):
        """A recorded knob that never reaches the optimizer is the failure this repo is built on."""
        config = make_config(
            adam_epsilon=1e-15, use_liger_kernel=False, output_dir="artifacts/games/runs/test"
        )
        args = gt._build_grpo_config(config, make_plan(), dtype=torch.float32)
        assert args.adam_epsilon == pytest.approx(1e-15)

    def test_an_unset_epsilon_reaches_trl_as_transformers_own_default(self):
        config = make_config(use_liger_kernel=False, output_dir="artifacts/games/runs/test")
        args = gt._build_grpo_config(config, make_plan(), dtype=torch.float32)
        assert args.adam_epsilon == pytest.approx(1e-8)

    def test_an_epsilon_of_zero_is_refused_before_a_card_is_reserved(self):
        with pytest.raises(ValueError, match="adam_epsilon"):
            make_config(adam_epsilon=0.0)

    def test_a_negative_epsilon_is_refused_too(self):
        with pytest.raises(ValueError, match="adam_epsilon"):
            make_config(adam_epsilon=-1e-15)

    @pytest.mark.parametrize("dropout", [1.0, 1.5, 2.0])
    def test_a_dropout_at_or_above_one_is_refused(self, dropout: float):
        """At 1.0 every adapter activation is dropped, so the graded forward is the base model."""
        with pytest.raises(ValueError, match="lora_dropout"):
            make_config(lora_dropout=dropout)

    @pytest.mark.parametrize("dropout", [-0.01, -1.0])
    def test_a_negative_dropout_is_refused(self, dropout: float):
        with pytest.raises(ValueError, match="lora_dropout"):
            make_config(lora_dropout=dropout)

    @pytest.mark.parametrize("dropout", [0.0, 0.05, 0.999])
    def test_the_open_interval_below_one_is_allowed(self, dropout: float):
        assert make_config(lora_dropout=dropout).lora_dropout == pytest.approx(dropout)

    @pytest.mark.parametrize("penalty", [0.0, 0.5])
    def test_a_parse_penalty_at_or_above_zero_is_refused_at_config_time(self, penalty: float):
        """The reward refuses it too, but only once a card is reserved and the weights are down."""
        with pytest.raises(ValueError, match="parse_penalty"):
            make_config(parse_penalty=penalty)

    @pytest.mark.parametrize("penalty", [float("nan"), float("-inf")])
    def test_a_non_finite_parse_penalty_is_refused_too(self, penalty: float):
        """`argparse` takes both for a `type=float`, and neither is a penalty.

        NaN is what TRL 1.10 reads as unscorable (`unscorable_mask` at grpo_trainer.py:2746-2754), so
        every unparseable completion would leave its group's baseline with a forced-zero advantage
        instead of being penalised, while `run_config.json` recorded a penalty. `-inf` sends the group
        mean to minus infinity and every advantage in it to NaN. `penalty >= 0` is False for both.

        Sabotage run once and watched red (2026-09-04): dropping the `math.isfinite` half of the
        refusal failed both cases here.
        """
        assert not penalty >= 0
        with pytest.raises(ValueError, match="finite negative number"):
            make_config(parse_penalty=penalty)

    def test_the_config_time_refusal_agrees_with_the_rewards_own(self):
        """Two copies of one rule, so the pair is asserted rather than trusted."""
        with pytest.raises(ValueError, match="parse_penalty"):
            gt.make_game_reward(8, prefilled_think=True, parse_penalty=0.0)
        make_config(parse_penalty=-0.001)
        gt.make_game_reward(8, prefilled_think=True, parse_penalty=-0.001)

    def test_both_are_resume_identity_fields(self):
        assert "adam_epsilon" in gt.RESUME_IDENTITY_FIELDS
        assert "lora_dropout" in gt.RESUME_IDENTITY_FIELDS

    def test_a_resume_under_a_different_epsilon_is_refused(self, tmp_path: Path):
        """The gate as `_prepare_run` calls it: the recorded launch's own `config` block against
        this launch's fields, so the refusal needs the value written down as well as watched."""
        recorded = gt.run_config_payload(
            make_config(adam_epsilon=1e-15, output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )["config"]
        assert isinstance(recorded, dict)
        with pytest.raises(RuntimeError, match="adam_epsilon"):
            gt.assert_resume_matches(
                recorded={**gt.RESUME_IDENTITY_DEFAULTS, **recorded},
                current=asdict(make_config(adam_epsilon=1e-8, output_dir=str(tmp_path / "run"))),
                fields=gt.RESUME_IDENTITY_FIELDS,
                checkpoint="checkpoint-70",
                consequence="two treatments under one set of step numbers.",
            )

    def test_a_resume_under_a_different_adapter_dropout_is_refused(self, tmp_path: Path):
        recorded = gt.run_config_payload(
            make_config(lora_dropout=0.0, output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )["config"]
        assert isinstance(recorded, dict)
        with pytest.raises(RuntimeError, match="lora_dropout"):
            gt.assert_resume_matches(
                recorded={**gt.RESUME_IDENTITY_DEFAULTS, **recorded},
                current=asdict(make_config(lora_dropout=0.05, output_dir=str(tmp_path / "run"))),
                fields=gt.RESUME_IDENTITY_FIELDS,
                checkpoint="checkpoint-70",
                consequence="two treatments under one set of step numbers.",
            )

    def test_a_resume_at_the_same_values_is_allowed(self, tmp_path: Path):
        config = make_config(adam_epsilon=1e-15, lora_dropout=0.0, output_dir=str(tmp_path / "run"))
        recorded = gt.run_config_payload(
            config, plan=make_plan(), device={"device_name": "NVIDIA L40S"}, derived={}
        )["config"]
        assert isinstance(recorded, dict)
        gt.assert_resume_matches(
            recorded={**gt.RESUME_IDENTITY_DEFAULTS, **recorded},
            current=asdict(config),
            fields=gt.RESUME_IDENTITY_FIELDS,
            checkpoint="checkpoint-70",
            consequence="two treatments under one set of step numbers.",
        )

    def test_a_record_predating_the_fields_resumes_under_the_values_its_era_ran(
        self, tmp_path: Path
    ):
        """Every banked run predates both fields, and absence must not read as a mismatch.

        The precedent is `RESUME_IDENTITY_DEFAULTS`, which maps a field absent from an old record to
        the value in force when that record was written: transformers' own 1e-8 for the epsilon,
        which nothing set, and 0.05 for the dropout, the default every banked arm carried.
        """
        assert gt.RESUME_IDENTITY_DEFAULTS["adam_epsilon"] == pytest.approx(1e-8)
        assert gt.RESUME_IDENTITY_DEFAULTS["lora_dropout"] == pytest.approx(0.05)
        recorded = gt.run_config_payload(
            make_config(output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )["config"]
        assert isinstance(recorded, dict)
        del recorded["adam_epsilon"]
        del recorded["lora_dropout"]
        gt.assert_resume_matches(
            recorded={**gt.RESUME_IDENTITY_DEFAULTS, **recorded},
            current=asdict(make_config(output_dir=str(tmp_path / "run"))),
            fields=gt.RESUME_IDENTITY_FIELDS,
            checkpoint="checkpoint-70",
            consequence="two treatments under one set of step numbers.",
        )

    def test_a_record_predating_the_fields_refuses_the_wave_values(self, tmp_path: Path):
        """The other half: an old checkpoint cannot be continued under the new treatment silently."""
        recorded = gt.run_config_payload(
            make_config(output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )["config"]
        assert isinstance(recorded, dict)
        del recorded["adam_epsilon"]
        with pytest.raises(RuntimeError, match="adam_epsilon"):
            gt.assert_resume_matches(
                recorded={**gt.RESUME_IDENTITY_DEFAULTS, **recorded},
                current=asdict(make_config(adam_epsilon=1e-15, output_dir=str(tmp_path / "run"))),
                fields=gt.RESUME_IDENTITY_FIELDS,
                checkpoint="checkpoint-70",
                consequence="two treatments under one set of step numbers.",
            )


class TestTheSamplerMismatchInstrumentsSurviveAResume:
    """The three knobs that decide WHICH ESTIMATOR TRL computes are resume identity, like loss_type.

    The shape this closes: the probe box runs the correction ON in log-only mode with the fp32 head,
    dies at step 3, and is relaunched with the training role's environment, where the correction is
    off. Steps 4 onward would then train under a different estimator -- with the correction on and
    log-only off the surrogate loss carries TRL's sequence-mask ratio, which zeroes whole sequences
    past its clip -- under one set of step numbers and one summary, and only a reader who diffed the
    two run records would ever know. `reward_hacking.train` pins the correction for this reason.

    Sabotage run once and watched red (2026-09-04): dropping
    `vllm_importance_sampling_log_only` from `RESUME_IDENTITY_FIELDS` failed three tests here.
    """

    @pytest.fixture
    def engine_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin vllm as installed, so a 32k correction-ON config builds at all in these tests."""
        monkeypatch.setattr(gt.importlib.util, "find_spec", lambda name: object())

    def recorded_config(self, config: gt.GameTrainConfig) -> dict[str, object]:
        """The `config` block of the run record a launch would have written."""
        recorded = gt.run_config_payload(
            config, plan=make_plan(), device={"device_name": "NVIDIA L40S"}, derived={}
        )["config"]
        assert isinstance(recorded, dict)
        return cast("dict[str, object]", recorded)

    def resume(self, recorded: dict[str, object], current: gt.GameTrainConfig) -> None:
        """The gate as `_prepare_run` calls it, over the record's own `config` block."""
        gt.assert_resume_matches(
            recorded={**gt.RESUME_IDENTITY_DEFAULTS, **recorded},
            current=asdict(current),
            fields=gt.RESUME_IDENTITY_FIELDS,
            checkpoint="checkpoint-3",
            consequence="two estimators under one set of step numbers.",
        )

    @pytest.mark.parametrize(
        "field",
        [
            "vllm_importance_sampling_correction",
            "vllm_importance_sampling_log_only",
            "vllm_importance_sampling_mode",
            "cast_lm_head_to_fp32",
        ],
    )
    def test_each_instrument_is_a_resume_identity_field(self, field: str):
        assert field in gt.RESUME_IDENTITY_FIELDS

    def test_changing_the_correction_mode_on_the_relaunch_is_refused(self, tmp_path: Path):
        """The mode is the estimator: sequence-level masking zeroes whole rollouts where token-level
        truncation clips each token, so steps taken under one continuing under the other would put
        two estimators in one step history with only a diff of two run records to show it."""
        recorded = self.recorded_config(
            make_config(
                vllm_importance_sampling_mode="token_truncate", output_dir=str(tmp_path / "run")
            )
        )
        with pytest.raises(RuntimeError, match="vllm_importance_sampling_mode"):
            self.resume(recorded, make_config(output_dir=str(tmp_path / "run")))

    def test_a_record_predating_the_mode_resumes_under_trls_default(self, tmp_path: Path):
        """Absence means the mode TRL had in force, which is the one every banked arm trained under."""
        assert (
            gt.RESUME_IDENTITY_DEFAULTS["vllm_importance_sampling_mode"]
            == VLLM_IMPORTANCE_SAMPLING_MODE
        )
        recorded = self.recorded_config(make_config(output_dir=str(tmp_path / "run")))
        del recorded["vllm_importance_sampling_mode"]
        self.resume(recorded, make_config(output_dir=str(tmp_path / "run")))

    def test_dropping_the_correction_on_the_relaunch_is_refused(self, tmp_path: Path):
        recorded = self.recorded_config(make_config(output_dir=str(tmp_path / "run")))
        with pytest.raises(RuntimeError, match="vllm_importance_sampling_correction"):
            self.resume(
                recorded,
                make_config(
                    vllm_importance_sampling_correction=False, output_dir=str(tmp_path / "run")
                ),
            )

    def test_turning_log_only_off_on_the_relaunch_is_refused(
        self, tmp_path: Path, engine_installed: None
    ):
        """Log-only off is the flip that starts weighting the gradient by the measured ratio."""
        del engine_installed
        recorded = self.recorded_config(
            make_config(vllm_importance_sampling_log_only=True, output_dir=str(tmp_path / "run"))
        )
        with pytest.raises(RuntimeError, match="vllm_importance_sampling_log_only"):
            self.resume(recorded, make_config(output_dir=str(tmp_path / "run")))

    def test_dropping_the_fp32_head_on_the_relaunch_is_refused(
        self, tmp_path: Path, engine_installed: None
    ):
        del engine_installed
        recorded = self.recorded_config(
            make_config(cast_lm_head_to_fp32=True, output_dir=str(tmp_path / "run"))
        )
        with pytest.raises(RuntimeError, match="cast_lm_head_to_fp32"):
            self.resume(recorded, make_config(output_dir=str(tmp_path / "run")))

    def test_a_relaunch_at_the_same_instruments_is_allowed(
        self, tmp_path: Path, engine_installed: None
    ):
        del engine_installed
        config = make_config(
            vllm_importance_sampling_log_only=True,
            cast_lm_head_to_fp32=True,
            output_dir=str(tmp_path / "run"),
        )
        self.resume(self.recorded_config(config), config)

    def test_a_record_predating_the_fields_resumes_under_what_its_era_ran(self, tmp_path: Path):
        """Every banked record predates the two knobs and carries TRL's correction default.

        Absence means THIS, not today's default and not a mismatch: the correction has been on since
        colocate landed, and log-only and the fp32 head did not exist, so a record without them
        continues as a correction-ON run with neither knob set.
        """
        assert gt.RESUME_IDENTITY_DEFAULTS["vllm_importance_sampling_correction"] is True
        assert gt.RESUME_IDENTITY_DEFAULTS["vllm_importance_sampling_log_only"] is False
        assert gt.RESUME_IDENTITY_DEFAULTS["cast_lm_head_to_fp32"] is False
        recorded = self.recorded_config(make_config(output_dir=str(tmp_path / "run")))
        for field in (
            "vllm_importance_sampling_correction",
            "vllm_importance_sampling_log_only",
            "cast_lm_head_to_fp32",
        ):
            del recorded[field]
        self.resume(recorded, make_config(output_dir=str(tmp_path / "run")))

    def test_a_record_predating_the_fields_still_refuses_the_probe_instruments(
        self, tmp_path: Path, engine_installed: None
    ):
        """The other half: an old checkpoint cannot be continued under the probe's estimator."""
        del engine_installed
        recorded = self.recorded_config(make_config(output_dir=str(tmp_path / "run")))
        del recorded["vllm_importance_sampling_log_only"]
        with pytest.raises(RuntimeError, match="vllm_importance_sampling_log_only"):
            self.resume(
                recorded,
                make_config(
                    vllm_importance_sampling_log_only=True, output_dir=str(tmp_path / "run")
                ),
            )


def checkpoint_config_stub(*, tied: bool) -> SimpleNamespace:
    """An `AutoConfig` stand-in answering only the tie flag, off the text sub-config TRL reads."""
    return SimpleNamespace(get_text_config=lambda: SimpleNamespace(tie_word_embeddings=tied))


class TestTheFp32HeadIsRefusedOnATiedCheckpoint:
    """TRL's own fp32-head cast crashes on a tied-embedding checkpoint under a PEFT adapter.

    `_cast_lm_head_to_fp32` runs on the PeftModel (`model` is reassigned to it at
    grpo_trainer.py:447, the cast is called at :1026) and its tied branch dereferences
    `target_model.model.embed_tokens` (:1024). One level below a PeftModel that resolves to the
    causal-LM wrapper, whose embeddings sit another level down, so the cast raises AttributeError
    inside `GRPOTrainer.__init__` -- after the weights are down on a rented card, and reading like a
    bug in our own plumbing. Reproduced on CPU this session with a tiny tied Qwen2 wrapped by
    `get_peft_model`: `pm.model.embed_tokens` -> AttributeError.

    Sabotage run once and watched red (2026-09-04): returning from
    `assert_cast_lm_head_is_supported` before it reads the tie flag failed three tests here.
    """

    @pytest.fixture(autouse=True)
    def engine_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The cast needs the correction ON, which needs vllm to look installed."""
        monkeypatch.setattr(gt.importlib.util, "find_spec", lambda name: object())

    def test_a_tied_checkpoint_is_refused_by_name(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            gt.AutoConfig, "from_pretrained", lambda _model: checkpoint_config_stub(tied=True)
        )
        with pytest.raises(ValueError, match="tie_word_embeddings=True"):
            gt.assert_cast_lm_head_is_supported("Qwen/Qwen3.5-4B", cast_lm_head_to_fp32=True)

    def test_the_refusal_names_the_trl_lines_and_the_rung_that_works(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            gt.AutoConfig, "from_pretrained", lambda _model: checkpoint_config_stub(tied=True)
        )
        with pytest.raises(ValueError, match=r"grpo_trainer\.py:1017-1026"):
            gt.assert_cast_lm_head_is_supported("Qwen/Qwen3.5-2B", cast_lm_head_to_fp32=True)
        with pytest.raises(ValueError, match=r"Qwen3\.5-9B"):
            gt.assert_cast_lm_head_is_supported("Qwen/Qwen3.5-2B", cast_lm_head_to_fp32=True)

    def test_an_untied_checkpoint_is_allowed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            gt.AutoConfig, "from_pretrained", lambda _model: checkpoint_config_stub(tied=False)
        )
        gt.assert_cast_lm_head_is_supported("Qwen/Qwen3.5-9B", cast_lm_head_to_fp32=True)

    def test_the_checkpoint_is_not_read_at_all_without_the_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Off by default, so the guard must cost nothing -- not even a hub read -- on every other run."""
        reads: list[str] = []

        def record(model_id: str) -> SimpleNamespace:
            reads.append(model_id)
            return checkpoint_config_stub(tied=True)

        monkeypatch.setattr(gt.AutoConfig, "from_pretrained", record)
        gt.assert_cast_lm_head_is_supported("Qwen/Qwen3.5-4B", cast_lm_head_to_fp32=False)
        assert reads == []

    def test_the_launch_refuses_before_it_resolves_anything(self, monkeypatch: pytest.MonkeyPatch):
        """Wired into `_announce_launch`, which runs before the resume resolve and the device read."""
        monkeypatch.setattr(
            gt.AutoConfig, "from_pretrained", lambda _model: checkpoint_config_stub(tied=True)
        )
        config = make_config(cast_lm_head_to_fp32=True)
        with pytest.raises(ValueError, match="tie_word_embeddings=True"):
            gt._announce_launch(config)  # pyright: ignore[reportPrivateUsage]

    def test_an_untied_launch_gets_past_the_guard(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            gt.AutoConfig, "from_pretrained", lambda _model: checkpoint_config_stub(tied=False)
        )
        assert gt._announce_launch(make_config(cast_lm_head_to_fp32=True)) == 1  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.skipif(
        not os.environ.get("RLVR_SMOKE"), reason="reads the real checkpoint configs (needs cache)"
    )
    @pytest.mark.parametrize(
        ("model_id", "tied"),
        [
            ("Qwen/Qwen3.5-2B", True),
            ("Qwen/Qwen3.5-4B", True),
            ("Qwen/Qwen3.5-9B", False),
            (gt.SMOKE_MODEL_ID, True),
        ],
    )
    def test_the_ladder_is_tied_everywhere_below_the_9b(self, model_id: str, tied: bool):
        """The claim the refusal message makes about this ladder, against the checkpoints themselves."""
        live = AutoConfig.from_pretrained(model_id).get_text_config().tie_word_embeddings
        assert bool(live) is tied


CARE_ARM_NAME = "care-under-test"


@pytest.fixture
def care_arm(monkeypatch: pytest.MonkeyPatch) -> GameArm:
    """Register one care arm for the duration of a test, so no test depends on the live registry."""
    arm = GameArm(
        game_id="twin-pd", grading=care_grading(1), notes="a care arm registered inside a test"
    )
    monkeypatch.setitem(gt.ARMS, CARE_ARM_NAME, arm)
    return arm


class TestTheRunRecordCarriesTheCareWeight:
    """The care family's weight is a number the readouts key on, so the record names it as one.

    `care-alpha-0` and `care-alpha-1` are wave 4b's pair, and a record carrying only the grading
    string would make the treatment a string match in every readout that compares them.
    """

    def test_a_care_arm_records_its_weight(self, tmp_path: Path, care_arm: GameArm) -> None:
        del care_arm
        payload = gt.run_config_payload(
            make_config(arm=CARE_ARM_NAME, output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )
        assert payload["grading"] == "care-alpha-1"
        assert payload["care_alpha"] == pytest.approx(1.0)

    def test_the_weight_survives_the_json_round_trip_as_a_number(
        self, tmp_path: Path, care_arm: GameArm
    ) -> None:
        del care_arm
        payload = gt.run_config_payload(
            make_config(arm=CARE_ARM_NAME, output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )
        restored = json.loads(json.dumps(payload, indent=2, default=str))
        assert restored["care_alpha"] == pytest.approx(1.0)

    def test_every_other_arm_records_no_weight(self, tmp_path: Path) -> None:
        payload = gt.run_config_payload(
            make_config(output_dir=str(tmp_path / "run")),
            plan=make_plan(),
            device={"device_name": "NVIDIA L40S"},
            derived={},
        )
        assert payload["care_alpha"] is None


class TestTheCareFamilysBehaviouralMetricsFollowTheCorpus:
    """A care corpus can hold both row types, so which metrics it must produce is a corpus question.

    The dictator arm's bug in a new shape: requiring only `coop_rate` would leave the trust sender's
    ONLY behavioural number -- the fraction of its stock it sent -- outside the read-back gate, and a
    run that never logged it would still write a clean-looking summary.
    """

    def test_a_matrix_only_care_corpus_requires_the_cooperation_rate_alone(self) -> None:
        required = gt.required_metrics_for(care_grading(1))
        assert "coop_rate" in required
        assert "mean_send_fraction" not in required

    def test_a_care_corpus_with_trust_rows_requires_the_send_fraction_too(self) -> None:
        required = gt.required_metrics_for(care_grading(1), announced_rule_trust_rows=True)
        assert "coop_rate" in required
        assert "mean_send_fraction" in required

    def test_a_care_run_that_never_logged_the_send_fraction_is_reported_as_missing(self) -> None:
        history = [healthy_record(1)]
        _, missing = gt.read_back_metrics(
            fake_metric_trainer(history),  # pyright: ignore[reportArgumentType]
            required=gt.required_metrics_for(care_grading(1), announced_rule_trust_rows=True),
            constant_by_construction=gt.constant_by_construction_metrics(
                gt.PARSE_PENALTY_MARGIN_BELOW_WORSE
            ),
        )
        assert missing == ["mean_send_fraction"]

    def test_the_flag_is_read_off_the_corpus_rather_than_the_arm(self) -> None:
        matrix_only = Dataset.from_list(
            [{"prompt_id": "matrix", gt.STATED_RETURN_FRACTION_COLUMN: STATED_RETURN_UNSET}]
        )
        mixed = Dataset.from_list(
            [
                {"prompt_id": "matrix", gt.STATED_RETURN_FRACTION_COLUMN: STATED_RETURN_UNSET},
                {"prompt_id": "trust", gt.STATED_RETURN_FRACTION_COLUMN: 0.5},
            ]
        )
        assert not gt.dataset_carries_announced_rule_trust_rows(matrix_only)
        assert gt.dataset_carries_announced_rule_trust_rows(mixed)

    def test_a_corpus_without_the_column_at_all_carries_no_trust_rows(self) -> None:
        """An older corpus swept before the column existed: absent is not a trust row."""
        without = Dataset.from_list([{"prompt_id": "matrix"}])
        assert not gt.dataset_carries_announced_rule_trust_rows(without)
