"""Dynamic sampling: over-generate prompt groups, train on the ones that carry a gradient.

A *pure* group is one whose completions all earned the same reward. GRPO's advantage is the reward
less the group mean, so every row of such a group is exactly zero: the group costs a full training
forward and backward and moves no parameter. On the banked 9B pair a fifth of groups were pure,
rising to a third late in the run, and on wave 4b's breadth corpus the far framings are expected to
go pure most of the time. `games.train.DynamicSampledGRPOTrainer` generates
`dynamic_sampling_oversample` times a step's groups, scores all of them, and hands the optimizer the
live ones.

Four things are pinned here, all on CPU:

*   the selection arithmetic, on fake advantage tensors: live groups first in sampler order, pure
    groups only as fallback, the batch always exactly the width TRL split it to expect;
*   the narrowing, on TRL-shaped batches: every per-row key indexed in step, the per-sequence ratio
    and the token-level ratio both, and `num_items_in_batch` RECOMPUTED rather than inherited, which
    is the difference between the kept batch's own gradient scale and half of it;
*   the trainer seam, with `super()._generate_and_score_completions` stubbed the way
    `test_games_trainer_instruments.py` stubs it: the metrics, the trace's dropped column, and
    oversample 1 handing back the identical object with no new metric at all;
*   the wiring: the config field, the flag, the run record, the resume identity, the widened
    generation batch, and the two gates that would otherwise read a wide generation as a broken
    trace.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import asdict, fields
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict

import games.train as gt
from games import preflight, sizing

logger = logging.getLogger(__name__)

NUM_GENERATIONS = 2


def make_config(**overrides: object) -> gt.GameTrainConfig:
    """A valid config, with whatever the test cares about overridden."""
    base: dict[str, object] = {"arm": "twin-pd-group", "generate_fresh": True}
    return gt.GameTrainConfig(**(base | overrides))  # pyright: ignore[reportArgumentType]


def hybrid_text_config() -> SimpleNamespace:
    """A Qwen3.5-shaped config: three linear-attention layers per full-attention layer."""
    return SimpleNamespace(
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        * 8,
        num_key_value_heads=4,
        head_dim=128,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )


def make_plan(**overrides: object) -> sizing.SizingPlan:
    """A sizing plan, without touching a GPU or a checkpoint."""
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


def advantages_for(*groups: bool) -> torch.Tensor:
    """One advantage row per completion: a live group disagrees, a pure group is all zeros."""
    rows: list[float] = []
    for live in groups:
        rows.extend([0.7, -0.7] if live else [0.0, 0.0])
    return torch.tensor(rows)


class TestChoosingTheLiveGroups:
    def test_a_live_group_is_kept_over_a_pure_one_whatever_order_they_arrived_in(self) -> None:
        selection = gt.choose_live_groups(
            advantages_for(False, True), num_generations=NUM_GENERATIONS, keep_groups=1
        )
        assert selection.live_groups == (1,)
        assert selection.kept_live == (1,)
        assert selection.kept_pure == ()
        assert selection.kept_groups == (1,)
        assert selection.groups_pure == 1

    def test_live_groups_are_taken_in_sampler_order_not_by_how_much_advantage_they_carry(
        self,
    ) -> None:
        """Keeping the biggest-spread groups would bias the batch toward whatever the reward happens
        to spread, which is the quantity these arms measure."""
        advantages = torch.tensor([0.1, -0.1, 5.0, -5.0, 0.2, -0.2])
        selection = gt.choose_live_groups(
            advantages, num_generations=NUM_GENERATIONS, keep_groups=2
        )
        assert selection.kept_groups == (0, 1)

    def test_pure_groups_fill_the_batch_when_too_few_live_ones_exist(self) -> None:
        """The batch handed back must be the width TRL split it to expect, so the fallback is not
        optional -- and it is counted separately, because a step that used it is a step the
        oversample was too narrow for."""
        selection = gt.choose_live_groups(
            advantages_for(False, True, False), num_generations=NUM_GENERATIONS, keep_groups=2
        )
        assert selection.kept_live == (1,)
        assert selection.kept_pure == (0,)
        assert selection.kept_groups == (0, 1)

    def test_a_batch_with_nothing_live_keeps_pure_groups_in_sampler_order(self) -> None:
        selection = gt.choose_live_groups(
            advantages_for(False, False, False), num_generations=NUM_GENERATIONS, keep_groups=2
        )
        assert selection.live_groups == ()
        assert selection.kept_live == ()
        assert selection.kept_pure == (0, 1)
        assert selection.groups_pure == 3

    def test_a_group_with_one_nonzero_row_counts_as_live(self) -> None:
        """TRL forces an unscorable row's advantage to zero (grpo_trainer.py:2799), so a group can
        be part zero and still supply a gradient."""
        selection = gt.choose_live_groups(
            torch.tensor([0.0, 0.4]), num_generations=NUM_GENERATIONS, keep_groups=1
        )
        assert selection.live_groups == (0,)

    def test_every_generated_group_is_accounted_for(self) -> None:
        selection = gt.choose_live_groups(
            advantages_for(True, False, True, False), num_generations=NUM_GENERATIONS, keep_groups=2
        )
        assert selection.groups_generated == 4
        assert len(selection.live_groups) + selection.groups_pure == 4

    def test_rows_that_are_not_whole_groups_are_refused(self) -> None:
        with pytest.raises(RuntimeError, match="whole number"):
            gt.choose_live_groups(torch.zeros(5), num_generations=NUM_GENERATIONS, keep_groups=1)

    def test_keeping_more_groups_than_were_generated_is_refused(self) -> None:
        """The dataloader width, the widened generation batch and the oversample have to agree; if
        they do not, the batch handed to TRL is short and the shapes fail inside the loss."""
        with pytest.raises(RuntimeError, match="short"):
            gt.choose_live_groups(
                advantages_for(True), num_generations=NUM_GENERATIONS, keep_groups=2
            )


def scored_batch(*groups: bool, token_level_ratio: bool = False) -> dict[str, Any]:
    """A TRL-shaped scored generation batch: two completions per group, one live token each."""
    rows = 2 * len(groups)
    completion_mask = torch.ones(rows, 3, dtype=torch.long)
    # One row is shorter, so a recomputed token count cannot coincide with a scaled one.
    completion_mask[0, 2] = 0
    batch: dict[str, Any] = {
        "prompt_ids": torch.arange(rows * 4).view(rows, 4),
        "prompt_mask": torch.ones(rows, 4, dtype=torch.long),
        "completion_ids": torch.arange(rows * 3).view(rows, 3),
        "completion_mask": completion_mask,
        "advantages": advantages_for(*groups),
        "num_items_in_batch": completion_mask.sum(),
        "sampling_per_token_logps": torch.randn(rows, 3),
        "importance_sampling_ratio": (
            torch.rand(rows, 3) if token_level_ratio else torch.rand(rows, 1)
        ),
    }
    return batch


class TestNarrowingTheBatchToTheKeptGroups:
    def test_every_per_row_key_is_indexed_in_step(self) -> None:
        batch = scored_batch(False, True)
        narrowed = gt.narrow_to_groups(batch, num_generations=NUM_GENERATIONS, groups=(1,))
        assert set(narrowed) == set(batch)
        for key in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask"):
            assert torch.equal(narrowed[key], batch[key][2:]), key
        assert torch.equal(narrowed["advantages"], batch["advantages"][2:])
        assert torch.equal(
            narrowed["sampling_per_token_logps"], batch["sampling_per_token_logps"][2:]
        )

    @pytest.mark.parametrize("token_level_ratio", [False, True])
    def test_the_importance_ratio_is_narrowed_at_either_granularity(
        self, token_level_ratio: bool
    ) -> None:
        """The sequence modes give a (B, 1) ratio and the token modes a (B, T) one; both are
        row-major, so both are indexed, and a mode-specific narrowing would silently misalign one."""
        batch = scored_batch(False, True, token_level_ratio=token_level_ratio)
        narrowed = gt.narrow_to_groups(batch, num_generations=NUM_GENERATIONS, groups=(1,))
        assert torch.equal(
            narrowed["importance_sampling_ratio"], batch["importance_sampling_ratio"][2:]
        )

    def test_the_live_token_count_is_recomputed_from_the_kept_rows(self) -> None:
        """The one value that is not per-row. TRL sums the loss mask of the WHOLE generation batch
        (grpo_trainer.py:2497) and the dapo, cispo and vespo normalizers divide by it (:3210-3214),
        so a narrowed batch that inherited the count would train at a fraction of the gradient scale
        with every logged series unchanged."""
        batch = scored_batch(False, True)
        narrowed = gt.narrow_to_groups(batch, num_generations=NUM_GENERATIONS, groups=(1,))
        assert int(narrowed["num_items_in_batch"]) == int(narrowed["completion_mask"].sum())
        assert int(narrowed["num_items_in_batch"]) < int(batch["num_items_in_batch"])

    def test_a_tool_mask_narrows_the_count_the_way_trl_narrows_it(self) -> None:
        batch = scored_batch(False, True)
        batch["tool_mask"] = torch.ones_like(batch["completion_mask"])
        batch["tool_mask"][2, 0] = 0
        narrowed = gt.narrow_to_groups(batch, num_generations=NUM_GENERATIONS, groups=(1,))
        expected = narrowed["completion_mask"] * narrowed["tool_mask"]
        assert int(narrowed["num_items_in_batch"]) == int(expected.sum())

    def test_keeping_every_group_is_the_batch_it_was_handed(self) -> None:
        batch = scored_batch(True, True)
        narrowed = gt.narrow_to_groups(batch, num_generations=NUM_GENERATIONS, groups=(0, 1))
        for key, value in batch.items():
            assert torch.equal(narrowed[key], value), key

    def test_a_key_that_is_neither_per_row_nor_a_scalar_is_refused(self) -> None:
        """A row-aligned tensor left at the generated width is a shape mismatch the loss meets first,
        so an unrecognised key raises here rather than passing through."""
        batch = scored_batch(True, True)
        batch["something_trl_added"] = torch.zeros(3, 3)
        with pytest.raises(RuntimeError, match="something_trl_added"):
            gt.narrow_to_groups(batch, num_generations=NUM_GENERATIONS, groups=(0,))

    def test_a_per_row_list_column_is_narrowed_too(self) -> None:
        batch = scored_batch(True, True)
        batch["num_images"] = [0, 0, 1, 1]
        narrowed = gt.narrow_to_groups(batch, num_generations=NUM_GENERATIONS, groups=(1,))
        assert narrowed["num_images"] == [1, 1]


def bare_trainer(
    monkeypatch: pytest.MonkeyPatch,
    batch: dict[str, Any],
    *,
    oversample: int,
    num_generations: int = NUM_GENERATIONS,
) -> gt.DynamicSampledGRPOTrainer:
    """The seam on a bare instance, with `GRPOTrainer._generate_and_score_completions` stubbed.

    The instrumented layers between this class and TRL read `model.training`, the log-only flag and
    the metrics dict, so the stand-in carries those; nothing here builds a model.
    """
    trainer = gt.DynamicSampledGRPOTrainer.__new__(gt.DynamicSampledGRPOTrainer)
    trainer.model = SimpleNamespace(training=True)  # pyright: ignore[reportAttributeAccessIssue]
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}  # pyright: ignore[reportAttributeAccessIssue]
    # Bounded exactly as TRL bounds it (`grpo_trainer.py:1069-1074`, maxlen=generation_batch_size,
    # which the widening has already made the GENERATED width), so a column longer than one
    # generation shows up here as a column that lost its own first rows rather than as a pass.
    generated_rows = int(batch["advantages"].size(0))
    trainer._logs = {"extra": defaultdict(lambda: deque(maxlen=generated_rows))}  # pyright: ignore[reportAttributeAccessIssue]
    trainer.num_generations = num_generations  # pyright: ignore[reportAttributeAccessIssue]
    trainer.num_iterations = 1  # pyright: ignore[reportAttributeAccessIssue]
    trainer.args = SimpleNamespace(steps_per_generation=1, gradient_accumulation_steps=1)  # pyright: ignore[reportAttributeAccessIssue]
    trainer.importance_sampling_log_only = False
    trainer.dynamic_sampling_oversample = oversample
    monkeypatch.setattr(
        gt.GRPOTrainer, "_generate_and_score_completions", lambda _self, _batch: batch
    )
    return trainer


def dropped_column(trainer: gt.DynamicSampledGRPOTrainer) -> list[bool]:
    """Read the trace's dropped flags off a bare trainer, in the order they were appended.

    Cast because TRL declares `_logs` as one dict holding both the per-row deques and the nested
    `extra` mapping, so the static type of `_logs["extra"]` is their union.
    """
    extra_columns = cast("dict[str, Any]", trainer._logs)["extra"]  # pyright: ignore[reportPrivateUsage]
    return list(extra_columns[gt.DYNAMIC_SAMPLING_DROPPED_COLUMN])


def has_dropped_column(trainer: gt.DynamicSampledGRPOTrainer) -> bool:
    """Whether the trace column exists at all, which at oversample 1 it must not."""
    return gt.DYNAMIC_SAMPLING_DROPPED_COLUMN in cast("dict[str, Any]", trainer._logs)["extra"]  # pyright: ignore[reportPrivateUsage]


class TestTheTrainerSeam:
    def test_one_live_group_among_two_is_what_the_optimizer_gets(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        batch = scored_batch(False, True)
        trainer = bare_trainer(monkeypatch, batch, oversample=2)
        with caplog.at_level(logging.INFO, logger="games.train"):
            kept = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert torch.equal(kept["advantages"], batch["advantages"][2:])
        assert kept["completion_mask"].size(0) == NUM_GENERATIONS
        assert "dynamic sampling: generated=2 pure=1 kept_live=1 kept_pure=0" in caplog.text

    def test_the_step_counts_land_in_trls_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        trainer = bare_trainer(monkeypatch, scored_batch(False, True), oversample=2)
        trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        metrics = trainer._metrics["train"]  # pyright: ignore[reportPrivateUsage]
        assert metrics[gt.DYNAMIC_SAMPLING_GENERATED_METRIC] == [2.0]
        assert metrics[gt.DYNAMIC_SAMPLING_PURE_METRIC] == [1.0]
        assert metrics[gt.DYNAMIC_SAMPLING_KEPT_LIVE_METRIC] == [1.0]
        assert metrics[gt.DYNAMIC_SAMPLING_KEPT_PURE_METRIC] == [0.0]

    def test_a_step_with_nothing_live_keeps_pure_groups_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The fallback is what keeps the batch the width TRL expects, and its count is the honest
        record that this step trained on groups carrying no gradient."""
        trainer = bare_trainer(monkeypatch, scored_batch(False, False), oversample=2)
        with caplog.at_level(logging.INFO, logger="games.train"):
            kept = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert not bool(kept["advantages"].any())
        assert trainer._metrics["train"][gt.DYNAMIC_SAMPLING_KEPT_PURE_METRIC] == [1.0]  # pyright: ignore[reportPrivateUsage]
        assert "kept_live=0 kept_pure=1" in caplog.text

    def test_the_kept_batch_is_always_the_width_trl_split_it_to_expect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for groups in ((True, True, True, True), (False, True, False, False), (False,) * 4):
            batch = scored_batch(*groups)
            trainer = bare_trainer(monkeypatch, batch, oversample=2)
            kept = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
            assert kept["advantages"].size(0) == 2 * NUM_GENERATIONS, groups

    def test_every_generated_row_is_marked_kept_or_dropped_for_the_trace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One value per GENERATED row, which is the length `_logs["prompt"]` has and therefore the
        length every column of `completions_<step>.parquet` must have."""
        trainer = bare_trainer(monkeypatch, scored_batch(False, True), oversample=2)
        trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert dropped_column(trainer) == [True, True, False, False]

    def test_a_second_step_leaves_exactly_one_generations_worth_of_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TRL builds the parquet from these buffers as one DataFrame, so the column has to be one
        generation long: a column that grew per step would either misalign the table or, being a
        bounded deque, quietly drop its own oldest rows."""
        trainer = bare_trainer(monkeypatch, scored_batch(True, False), oversample=2)
        trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert dropped_column(trainer) == [False, False, True, True]

    def test_at_oversample_one_the_batch_and_the_metrics_are_untouched(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The default has to be today's behaviour exactly: the same object back, no selection, no
        new metric key, and nothing in the trace's extra columns."""
        batch = scored_batch(False, True)
        trainer = bare_trainer(monkeypatch, batch, oversample=1)
        with caplog.at_level(logging.INFO, logger="games.train"):
            kept = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert kept is batch
        assert not set(gt.DYNAMIC_SAMPLING_STEP_METRICS) & set(trainer._metrics["train"])  # pyright: ignore[reportPrivateUsage]
        assert not has_dropped_column(trainer)
        assert "dynamic sampling" not in caplog.text

    def test_the_dataloader_is_built_at_the_oversampled_width_and_the_attribute_restored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TRL sizes its training batch `_train_batch_size * steps_per_generation`
        (`grpo_trainer.py:1243`), the narrow width, so the override scales that factor for the length
        of the call. Without this the sampler chunks wide prompts while the dataloader hands one
        narrow batch at a time, and generation would see a different prompt set every step.

        The restore is half the check: `_train_batch_size` is also what transformers reports as the
        run's batch size and what `auto_find_batch_size` rewrites.
        """
        widths: list[int] = []
        trainer = gt.DynamicSampledGRPOTrainer.__new__(gt.DynamicSampledGRPOTrainer)
        trainer._train_batch_size = 4  # pyright: ignore[reportAttributeAccessIssue]
        trainer.dynamic_sampling_oversample = 3
        monkeypatch.setattr(
            gt.GRPOTrainer,
            "get_train_dataloader",
            lambda bound: widths.append(bound._train_batch_size),  # pyright: ignore[reportPrivateUsage]
        )
        trainer.get_train_dataloader()
        assert widths == [12]
        assert trainer._train_batch_size == 4  # pyright: ignore[reportPrivateUsage]

    def test_the_dataloader_width_is_untouched_at_oversample_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widths: list[int] = []
        trainer = gt.DynamicSampledGRPOTrainer.__new__(gt.DynamicSampledGRPOTrainer)
        trainer._train_batch_size = 4  # pyright: ignore[reportAttributeAccessIssue]
        trainer.dynamic_sampling_oversample = 1
        monkeypatch.setattr(
            gt.GRPOTrainer,
            "get_train_dataloader",
            lambda bound: widths.append(bound._train_batch_size),  # pyright: ignore[reportPrivateUsage]
        )
        trainer.get_train_dataloader()
        assert widths == [4]

    def test_the_kept_batch_survives_trls_own_shuffle_and_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The contract the width exists for, checked against TRL's own functions rather than
        restated: `_prepare_inputs` shuffles the returned batch and splits it into
        `steps_per_generation` micro-batches (`grpo_trainer.py:1588-1592`), and `split_tensor_dict`
        divides the first dimension by that count, so a batch of the wrong width silently yields
        micro-batches of the wrong size -- or, once it is not divisible, a shorter last one.
        """
        steps_per_generation = 2
        trainer = bare_trainer(monkeypatch, scored_batch(False, True, True, False), oversample=2)
        kept = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        # Casts only: TRL annotates these two helpers for narrower dicts than the batch it hands
        # them itself (`grpo_trainer.py:1588-1592` passes exactly this dict).
        shuffled = shuffle_sequence_dict(cast("dict[str, Any]", kept))
        micro_batches = split_tensor_dict(cast("dict[str, Any]", shuffled), steps_per_generation)
        assert len(micro_batches) == steps_per_generation
        for micro_batch in micro_batches:
            assert cast("torch.Tensor", micro_batch["completion_ids"]).size(0) == (
                2 * NUM_GENERATIONS // steps_per_generation
            )
            # The scalar rides along untouched, which is why it has to be right before the split.
            assert torch.equal(
                cast("torch.Tensor", micro_batch["num_items_in_batch"]),
                kept["num_items_in_batch"],
            )

    def test_an_oversample_below_one_is_refused_by_the_trainer_too(self) -> None:
        """The config refuses it at launch; the class refuses it as well, because
        `reward_hacking.train` and the tests construct these trainers directly."""
        with pytest.raises(ValueError, match="dynamic_sampling_oversample"):
            gt.DynamicSampledGRPOTrainer(dynamic_sampling_oversample=0)


class TestTheClassesBelowItAreUnchanged:
    def test_the_padding_trimmed_trainer_is_still_what_reward_hacking_builds(self) -> None:
        """`reward_hacking.train` constructs `PaddingTrimmedGRPOTrainer` directly, so dynamic
        sampling has to be a class it does not build rather than a branch inside that one."""
        assert issubclass(gt.DynamicSampledGRPOTrainer, gt.PaddingTrimmedGRPOTrainer)
        assert not hasattr(gt.PaddingTrimmedGRPOTrainer, "dynamic_sampling_oversample")

    def test_the_class_default_is_off(self) -> None:
        assert gt.DynamicSampledGRPOTrainer.dynamic_sampling_oversample == 1


class TestTheWiring:
    OUTPUT_DIR = "artifacts/games/runs/test"

    def test_the_config_defaults_to_off_and_records_the_value(self) -> None:
        defaults = {field.name: field.default for field in fields(gt.GameTrainConfig)}
        assert defaults["dynamic_sampling_oversample"] == 1
        assert asdict(make_config())["dynamic_sampling_oversample"] == 1
        assert (
            asdict(make_config(dynamic_sampling_oversample=3))["dynamic_sampling_oversample"] == 3
        )

    def test_the_flag_is_spelled_the_way_the_kit_will_spell_it(self) -> None:
        config = gt._parse_args(  # pyright: ignore[reportPrivateUsage]
            ["--arm", "twin-pd-group", "--generate-fresh", "--dynamic-sampling-oversample", "2"]
        )
        assert config.dynamic_sampling_oversample == 2

    def test_the_command_line_default_matches_the_dataclass_default(self) -> None:
        config = gt._parse_args(["--arm", "twin-pd-group", "--generate-fresh"])  # pyright: ignore[reportPrivateUsage]
        assert config.dynamic_sampling_oversample == 1

    @pytest.mark.parametrize("oversample", [0, -1])
    def test_an_oversample_below_one_is_refused_at_startup(self, oversample: int) -> None:
        with pytest.raises(ValueError, match="dynamic_sampling_oversample"):
            make_config(dynamic_sampling_oversample=oversample)

    def test_it_is_a_resume_identity_field_with_off_as_the_era_default(self) -> None:
        assert "dynamic_sampling_oversample" in gt.RESUME_IDENTITY_FIELDS
        assert gt.RESUME_IDENTITY_DEFAULTS["dynamic_sampling_oversample"] == 1

    def test_a_resume_under_a_different_oversample_is_refused(self) -> None:
        """A different oversample is a different treatment AND a different walk through the corpus,
        so the resumed steps would neither train the same selection nor see the same prompts."""
        recorded = cast(
            "dict[str, object]",
            gt.run_config_payload(
                make_config(dynamic_sampling_oversample=2, output_dir=self.OUTPUT_DIR),
                plan=make_plan(),
                device={"device_name": "NVIDIA L40S"},
                derived={},
            )["config"],
        )
        with pytest.raises(RuntimeError, match="dynamic_sampling_oversample"):
            gt.assert_resume_matches(
                recorded={**gt.RESUME_IDENTITY_DEFAULTS, **recorded},
                current=asdict(make_config(output_dir=self.OUTPUT_DIR)),
                fields=gt.RESUME_IDENTITY_FIELDS,
                checkpoint="checkpoint-10",
                consequence="two selections under one set of step numbers.",
            )

    def test_a_record_predating_the_field_resumes_as_the_off_run_it_was(self) -> None:
        recorded = cast(
            "dict[str, object]",
            gt.run_config_payload(
                make_config(output_dir=self.OUTPUT_DIR),
                plan=make_plan(),
                device={"device_name": "NVIDIA L40S"},
                derived={},
            )["config"],
        )
        del recorded["dynamic_sampling_oversample"]
        gt.assert_resume_matches(
            recorded={**gt.RESUME_IDENTITY_DEFAULTS, **recorded},
            current=asdict(make_config(output_dir=self.OUTPUT_DIR)),
            fields=gt.RESUME_IDENTITY_FIELDS,
            checkpoint="checkpoint-10",
            consequence="two selections under one set of step numbers.",
        )

    def test_the_step_metrics_reach_the_summary_and_the_memory_log(self) -> None:
        assert set(gt.DYNAMIC_SAMPLING_STEP_METRICS) <= set(gt.OPTIONAL_METRICS)
        assert set(gt.DYNAMIC_SAMPLING_STEP_METRICS) <= set(gt.MEM_LOG_EXTRA_COLUMNS)

    def test_the_generation_batch_is_widened_and_the_split_is_not(self) -> None:
        """The two halves of TRL's derivation do different jobs: the batch size sizes the sampler's
        chunk and the completions buffer, while steps_per_generation decides how often TRL generates
        and into how many micro-batches it splits what comes back."""
        plan = make_plan()
        narrow = gt._build_grpo_config(  # pyright: ignore[reportPrivateUsage]
            make_config(output_dir=self.OUTPUT_DIR, use_liger_kernel=False),
            plan,
            dtype=torch.float32,
        )
        wide = gt._build_grpo_config(  # pyright: ignore[reportPrivateUsage]
            make_config(
                output_dir=self.OUTPUT_DIR, use_liger_kernel=False, dynamic_sampling_oversample=3
            ),
            plan,
            dtype=torch.float32,
        )
        assert narrow.generation_batch_size == plan.episodes_per_step
        assert wide.generation_batch_size == 3 * plan.episodes_per_step
        assert wide.steps_per_generation == narrow.steps_per_generation
        assert wide.gradient_accumulation_steps == narrow.gradient_accumulation_steps
        assert wide.per_device_train_batch_size == narrow.per_device_train_batch_size

    def test_the_trace_check_reads_the_wide_generation_as_complete(self) -> None:
        """Otherwise `check_built_trainer` raises on every wave-4b launch at build time, and
        `verify_trace_files` would call every step's parquet the wrong row count at the end of one."""
        assert preflight.check_trace_completeness(
            generation_batch_size=64,
            logging_steps=1,
            episodes_per_step=64,
            dynamic_sampling_oversample=1,
        )
        assert preflight.check_trace_completeness(
            generation_batch_size=192,
            logging_steps=1,
            episodes_per_step=64,
            dynamic_sampling_oversample=3,
        )

    def test_a_buffer_one_step_wide_under_an_oversample_is_still_refused(self) -> None:
        """The check keeps its teeth: a narrow buffer under a wide generation would keep only the
        tail of each generation, which is exactly the silent trace loss it exists for."""
        with pytest.raises(RuntimeError, match="buffered_episodes"):
            preflight.check_trace_completeness(
                generation_batch_size=64,
                logging_steps=1,
                episodes_per_step=64,
                dynamic_sampling_oversample=3,
            )

    def test_the_dataset_gate_asks_for_the_prompts_a_wide_generation_draws(self) -> None:
        """TRL's sampler chunks by the WIDENED generation batch and discards an incomplete chunk, so
        a corpus that fills a step at oversample 1 and not at 2 would train nothing at all."""
        plan = make_plan(prompts_per_step=8, num_generations=8)
        gt.assert_dataset_fills_a_step(8, plan)
        with pytest.raises(RuntimeError, match="one generation needs 16"):
            gt.assert_dataset_fills_a_step(8, plan, dynamic_sampling_oversample=2)
        gt.assert_dataset_fills_a_step(16, plan, dynamic_sampling_oversample=2)

    def test_the_banner_names_the_oversample_so_a_box_log_shows_the_treatment(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="games.train"):
            gt.log_colocate_settings(
                make_config(dynamic_sampling_oversample=2),
                engine_reserved_gib=33.2,
                old_logps_chunk_tokens=2048,
                old_logps_peak_gib=3.79,
                importance_sampling_log_only=False,
                dynamic_sampling_oversample=2,
            )
        assert "dynamic sampling is ON at oversample 2" in caplog.text

    def test_the_banner_says_nothing_about_it_when_it_is_off(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="games.train"):
            gt.log_colocate_settings(
                make_config(),
                engine_reserved_gib=33.2,
                old_logps_chunk_tokens=2048,
                old_logps_peak_gib=3.79,
                importance_sampling_log_only=False,
            )
        assert "dynamic sampling" not in caplog.text
