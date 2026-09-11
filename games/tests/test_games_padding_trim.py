"""The per-micro-batch padding trim: the slicing on fake batches, the trainer seam, and neutrality.

TRL 1.10 pads every completion in a generation batch to that batch's longest and splits the buffer
along rows only, so a micro-batch of one 5k-token completion trains as a 33k-token row whenever any
completion in the batch hit the cap. `games.train.trim_micro_batch` cuts each micro-batch to the
columns some row of it uses; `PaddingTrimmedGRPOTrainer` applies it at TRL's `_prepare_inputs`
seam. Three things are pinned here:

* the slicing itself, on CPU fake batches: left prompt pad, right completion pad, every per-token
  completion key sliced in step, non-token keys untouched, an all-zero mask left alone, a
  mis-shaped per-token key refused, and never a live token lost;
* the seam: the override trims what `super()._prepare_inputs` returns in training mode, leaves eval
  batches alone, and records padded/trimmed token counts into TRL's `_metrics`;
* neutrality, on the GPU: the padded and the trimmed pass of one real corpus row through the
  production loss path (`compute_liger_loss`, LoRA on the discovered targets, Liger fused GRPO
  loss, gradient checkpointing) give the same LoRA gradient (cosine > 0.999, norm within 1%), and
  the assertion has teeth: trimming one REAL token breaks it. Opt-in via GAMES_GPU_TESTS=1 because
  the only local card is shared and every GPU job here goes through gpu_preflight + the limiter.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

import games.train as gt
from games.prompts import generate_prompt_rows
from games.track_record_corpus import SOURCE_GAME_ID, build_track_record_v2_rows
from games.train import (
    COMPLETION_TOKEN_KEYS,
    PROMPT_TOKEN_KEYS,
    MicroBatchTrim,
    PaddingTrimmedGRPOTrainer,
    trim_micro_batch,
)
from grpo.throughput import discover_lora_targets

logger = logging.getLogger(__name__)


def fake_batch(  # noqa: PLR0913 - one keyword per padding axis of the batch being faked
    *,
    rows: int = 1,
    prompt_live: int = 5,
    prompt_left_pad: int = 3,
    completion_live: int = 4,
    completion_right_pad: int = 6,
    per_token_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """A TRL-shaped micro-batch: left-padded prompt, right-padded completion, optional extras."""
    prompt_width = prompt_left_pad + prompt_live
    completion_width = completion_live + completion_right_pad
    prompt_mask = torch.zeros(rows, prompt_width, dtype=torch.long)
    prompt_mask[:, prompt_left_pad:] = 1
    completion_mask = torch.zeros(rows, completion_width, dtype=torch.long)
    completion_mask[:, :completion_live] = 1
    batch: dict[str, Any] = {
        "prompt_ids": torch.arange(1, prompt_width + 1).repeat(rows, 1) * prompt_mask,
        "prompt_mask": prompt_mask,
        "completion_ids": torch.arange(101, completion_width + 101).repeat(rows, 1)
        * completion_mask,
        "completion_mask": completion_mask,
        "advantages": torch.full((rows,), 0.7),
        "num_items_in_batch": torch.tensor(64),
    }
    for key in per_token_keys:
        batch[key] = torch.randn(rows, completion_width)
    return batch


class TestTrimMicroBatch:
    def test_left_prompt_pad_and_right_completion_pad_are_cut(self) -> None:
        trimmed, stats = trim_micro_batch(fake_batch())
        assert trimmed["prompt_ids"].shape == (1, 5)
        assert trimmed["prompt_mask"].tolist() == [[1, 1, 1, 1, 1]]
        assert trimmed["prompt_ids"].tolist() == [[4, 5, 6, 7, 8]]
        assert trimmed["completion_ids"].shape == (1, 4)
        assert trimmed["completion_mask"].tolist() == [[1, 1, 1, 1]]
        assert trimmed["completion_ids"].tolist() == [[101, 102, 103, 104]]
        assert stats == MicroBatchTrim(rows=1, padded_columns=18, trimmed_columns=9)
        assert stats.padded_tokens == 18
        assert stats.trimmed_tokens == 9

    def test_no_live_token_is_ever_lost(self) -> None:
        batch = fake_batch(
            prompt_live=7, prompt_left_pad=11, completion_live=13, completion_right_pad=2
        )
        trimmed, _ = trim_micro_batch(batch)
        assert int(trimmed["prompt_mask"].sum()) == int(batch["prompt_mask"].sum())
        assert int(trimmed["completion_mask"].sum()) == int(batch["completion_mask"].sum())
        assert trimmed["prompt_mask"].all()
        assert trimmed["completion_mask"].all()

    def test_every_per_token_completion_key_is_sliced_in_step(self) -> None:
        extras = tuple(
            key for key in COMPLETION_TOKEN_KEYS if key not in ("completion_ids", "completion_mask")
        )
        batch = fake_batch(per_token_keys=extras)
        trimmed, _ = trim_micro_batch(batch)
        for key in extras:
            assert trimmed[key].shape == (1, 4), key
            assert torch.equal(trimmed[key], batch[key][:, :4]), key

    def test_a_per_sequence_importance_ratio_column_passes_through_untouched(self) -> None:
        """Under TRL's default `sequence_mask` importance-sampling mode the ratio is (B, 1), a value
        that broadcasts over the tokens rather than a padded row. The first L4 smoke with the
        correction ON died here: the width check read 1 against a 1,868-wide completion mask and
        raised on the first micro-batch. Sabotage: restoring the unconditional width check goes red."""
        batch = fake_batch(rows=2, per_token_keys=("sampling_per_token_logps",))
        batch["importance_sampling_ratio"] = torch.tensor([[0.9], [1.1]])
        trimmed, stats = trim_micro_batch(batch)
        assert trimmed["importance_sampling_ratio"] is batch["importance_sampling_ratio"]
        assert trimmed["sampling_per_token_logps"].shape == (2, 4)
        assert stats.trimmed_columns == 5 + 4

    def test_non_token_keys_pass_through_untouched(self) -> None:
        batch = fake_batch()
        batch["pixel_values"] = torch.zeros(1, 3, 8, 8)
        trimmed, _ = trim_micro_batch(batch)
        assert trimmed["advantages"] is batch["advantages"]
        assert trimmed["num_items_in_batch"] is batch["num_items_in_batch"]
        assert trimmed["pixel_values"] is batch["pixel_values"]
        assert set(trimmed) == set(batch)

    def test_an_absent_per_token_key_is_not_invented(self) -> None:
        trimmed, _ = trim_micro_batch(fake_batch())
        assert "old_per_token_logps" not in trimmed

    def test_an_all_zero_completion_mask_is_left_untrimmed(self) -> None:
        """No live column is a fact for the loss to see, not a width to invent."""
        batch = fake_batch(completion_live=0, completion_right_pad=6)
        trimmed, stats = trim_micro_batch(batch)
        assert trimmed["completion_ids"].shape == (1, 6)
        assert stats.trimmed_columns == 5 + 6

    def test_an_all_zero_prompt_mask_is_left_untrimmed(self) -> None:
        batch = fake_batch(prompt_live=0, prompt_left_pad=3)
        trimmed, stats = trim_micro_batch(batch)
        assert trimmed["prompt_ids"].shape == (1, 3)
        assert stats.trimmed_columns == 3 + 4

    def test_a_two_row_micro_batch_keeps_every_row_live_column(self) -> None:
        """Only columns that are pad in EVERY row go; a column live in one row stays for both."""
        batch = fake_batch(rows=2, completion_live=4, completion_right_pad=6)
        batch["completion_mask"][1, :7] = 1
        trimmed, stats = trim_micro_batch(batch)
        assert trimmed["completion_ids"].shape == (2, 7)
        assert stats.rows == 2
        assert stats.padded_tokens == 2 * 18
        assert stats.trimmed_tokens == 2 * (5 + 7)

    def test_a_mis_shaped_per_token_key_is_refused(self) -> None:
        batch = fake_batch()
        batch["old_per_token_logps"] = torch.randn(1, 3)
        with pytest.raises(RuntimeError, match=r"old_per_token_logps.*width 3"):
            trim_micro_batch(batch)

    def test_a_one_column_key_other_than_the_importance_ratio_is_still_refused(self) -> None:
        """The per-sequence exemption is the ratio's alone. A (B, 1) `old_per_token_logps` is exactly
        the mis-shape the width check exists to catch; exempting every one-column tensor (the first
        cut of the fix) would have let it broadcast over the tokens instead of raising."""
        batch = fake_batch()
        batch["old_per_token_logps"] = torch.randn(1, 1)
        with pytest.raises(RuntimeError, match=r"old_per_token_logps.*width 1"):
            trim_micro_batch(batch)
        assert {"importance_sampling_ratio"} == gt.PER_SEQUENCE_COMPLETION_KEYS

    def test_the_key_lists_name_what_trl_1_10_pads(self) -> None:
        """Pins the private layout the override depends on; a TRL upgrade that renames a key must
        surface here, not as a silently untrimmed column."""
        assert PROMPT_TOKEN_KEYS == ("prompt_ids", "prompt_mask")
        assert COMPLETION_TOKEN_KEYS[:2] == ("completion_ids", "completion_mask")
        assert {"old_per_token_logps", "ref_per_token_logps", "importance_sampling_ratio"} <= set(
            COMPLETION_TOKEN_KEYS
        )


class TestTheTrainerSeam:
    """The override on a bare instance: `super()._prepare_inputs` is stubbed to hand back a batch."""

    def bare_trainer(
        self, monkeypatch: pytest.MonkeyPatch, batch: dict[str, Any], *, training: bool
    ):
        trainer = PaddingTrimmedGRPOTrainer.__new__(PaddingTrimmedGRPOTrainer)
        trainer.model = SimpleNamespace(training=training)  # pyright: ignore[reportAttributeAccessIssue]
        trainer._step = 0  # pyright: ignore[reportAttributeAccessIssue]
        trainer.args = SimpleNamespace(steps_per_generation=2)  # pyright: ignore[reportAttributeAccessIssue]
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}  # pyright: ignore[reportAttributeAccessIssue]
        monkeypatch.setattr(gt.GRPOTrainer, "_prepare_inputs", lambda _self, _batch: batch)
        return trainer

    def test_a_training_micro_batch_comes_back_trimmed_and_counted(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        trainer = self.bare_trainer(monkeypatch, fake_batch(), training=True)
        out = trainer._prepare_inputs({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert out["completion_ids"].shape == (1, 4)
        assert trainer._metrics["train"]["padding_trim/padded_tokens"] == [18.0]  # pyright: ignore[reportPrivateUsage]
        assert trainer._metrics["train"]["padding_trim/trimmed_tokens"] == [9.0]  # pyright: ignore[reportPrivateUsage]
        # The step summary logs on the LAST micro-batch of the generation batch, not the first.
        assert "padding trim: step tokens" not in caplog.text
        assert gt.STEP_PADDED_TOKENS_METRIC not in trainer._metrics["train"]  # pyright: ignore[reportPrivateUsage]
        trainer._step = 1  # pyright: ignore[reportAttributeAccessIssue]
        with caplog.at_level("INFO", logger="games.train"):
            trainer._prepare_inputs({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert "padding trim: step tokens padded=36 trimmed=18 kept_fraction=0.500" in caplog.text
        # The step totals land in the same log record as the phase timer's seconds, once per step.
        assert trainer._metrics["train"][gt.STEP_PADDED_TOKENS_METRIC] == [36.0]  # pyright: ignore[reportPrivateUsage]
        assert trainer._metrics["train"][gt.STEP_TRIMMED_TOKENS_METRIC] == [18.0]  # pyright: ignore[reportPrivateUsage]
        assert set(gt.PADDING_TRIM_STEP_METRICS) <= set(gt.MEM_LOG_EXTRA_COLUMNS)

    def test_an_eval_batch_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        batch = fake_batch()
        trainer = self.bare_trainer(monkeypatch, batch, training=False)
        out = trainer._prepare_inputs({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert out is batch
        assert not trainer._metrics["train"]  # pyright: ignore[reportPrivateUsage]


GPU_TESTS_ENV = "GAMES_GPU_TESTS"


@pytest.mark.skipif(
    os.environ.get(GPU_TESTS_ENV) != "1" or not torch.cuda.is_available(),
    reason=f"GPU neutrality check: set {GPU_TESTS_ENV}=1 on a box with a free CUDA card (preflight + limiter)",
)
class TestNeutralityOnTheProductionLossPath:
    """Padded vs trimmed pass of one real corpus row, LoRA gradient compared.

    Builds the trainer exactly as `games.train` does minus the vLLM engine (Liger fused GRPO loss,
    LoRA on `discover_lora_targets`, gradient checkpointing, dapo/batch, micro-batch 1), zeroes LoRA
    dropout so the two passes see the same network, and compares the trainable gradients.
    """

    MODEL_ID = "Qwen/Qwen3.5-2B"
    COMPLETION_LIVE = 1536
    COMPLETION_PADDED = 4096
    PROMPT_LEFT_PAD = 64

    @pytest.fixture(scope="class")
    def rig(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        targets = cast("list[str]", discover_lora_targets(self.MODEL_ID)["target_modules"])
        args = GRPOConfig(
            output_dir=str(tmp_path_factory.mktemp("padding-trim-neutrality")),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,
            num_generations=8,
            max_completion_length=self.COMPLETION_PADDED,
            bf16=True,
            use_liger_kernel=True,
            gradient_checkpointing=True,
            beta=0.0,
            epsilon=0.2,
            loss_type="dapo",
            scale_rewards="batch",
            temperature=1.0,
            report_to="none",
            logging_steps=1,
            use_vllm=False,
            model_init_kwargs={"dtype": torch.bfloat16, "trust_remote_code": True},
        )
        trainer = GRPOTrainer(
            model=self.MODEL_ID,
            reward_funcs=lambda completions, **_: [0.0] * len(completions),
            args=args,
            train_dataset=Dataset.from_dict({"prompt": [[{"role": "user", "content": "hi"}]] * 8}),
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(targets),
            ),
        )
        model = cast("torch.nn.Module", trainer.model)
        model.train()
        unwrapped = trainer.accelerator.unwrap_model(model)
        unwrapped.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs
        )
        trainer.current_gradient_accumulation_steps = 64
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        # One real corpus row: a v2 chain-grid prompt built the way the corpus builder builds it.
        source = generate_prompt_rows(SOURCE_GAME_ID, "group-mix", split="train")
        one_reskin = [row for row in source if row["reskin_id"] == source[0]["reskin_id"]]
        corpus_rows, _ = build_track_record_v2_rows(one_reskin)
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": corpus_rows[0]["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_ids = torch.tensor(tokenizer(templated, add_special_tokens=False)["input_ids"])[None]
        return {
            "trainer": trainer,
            "unwrapped": unwrapped,
            "tokenizer": tokenizer,
            "prompt_ids": prompt_ids,
        }

    def padded_inputs(
        self, rig: dict[str, Any], *, completion_live: int
    ) -> dict[str, torch.Tensor]:
        device = rig["trainer"].accelerator.device
        pad_id = rig["tokenizer"].pad_token_id
        prompt_ids = rig["prompt_ids"]
        prompt_ids = torch.cat([torch.full((1, self.PROMPT_LEFT_PAD), pad_id), prompt_ids], dim=1)
        prompt_mask = torch.ones_like(prompt_ids)
        prompt_mask[:, : self.PROMPT_LEFT_PAD] = 0
        generator = torch.Generator().manual_seed(7)
        completion_ids = torch.randint(
            1000, 100000, (1, self.COMPLETION_PADDED), generator=generator
        )
        completion_mask = torch.zeros_like(completion_ids)
        completion_mask[:, :completion_live] = 1
        completion_ids[:, completion_live:] = pad_id
        return {
            "prompt_ids": prompt_ids.to(device),
            "prompt_mask": prompt_mask.to(device),
            "completion_ids": completion_ids.to(device),
            "completion_mask": completion_mask.to(device),
            "advantages": torch.tensor([0.7], device=device),
        }

    def lora_gradient(
        self, rig: dict[str, Any], inputs: dict[str, torch.Tensor]
    ) -> tuple[float, torch.Tensor]:
        trainer, model = rig["trainer"], rig["trainer"].model
        loss = trainer.compute_liger_loss(rig["unwrapped"], inputs)
        loss.backward()
        flat = torch.cat(
            [
                p.grad.float().flatten()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
        )
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return float(loss.detach().float()), flat

    def test_the_trimmed_pass_matches_the_padded_pass(self, rig: dict[str, Any]) -> None:
        padded = self.padded_inputs(rig, completion_live=self.COMPLETION_LIVE)
        trimmed, stats = trim_micro_batch(padded)
        assert (
            stats.trimmed_columns
            == padded["prompt_ids"].size(1) - self.PROMPT_LEFT_PAD + self.COMPLETION_LIVE
        )
        loss_padded, grad_padded = self.lora_gradient(rig, padded)
        loss_trimmed, grad_trimmed = self.lora_gradient(rig, trimmed)
        cosine = torch.nn.functional.cosine_similarity(grad_padded, grad_trimmed, dim=0).item()
        norm_ratio = (grad_trimmed.norm() / grad_padded.norm()).item()
        logger.info(
            "NEUTRALITY loss_padded=%.6f loss_trimmed=%.6f cosine=%.6f norm_ratio=%.6f "
            "padded_columns=%d trimmed_columns=%d",
            loss_padded,
            loss_trimmed,
            cosine,
            norm_ratio,
            stats.padded_columns,
            stats.trimmed_columns,
        )
        assert loss_trimmed == pytest.approx(loss_padded, rel=1e-3, abs=1e-6)
        assert cosine > 0.999
        assert abs(norm_ratio - 1.0) < 0.01

    def test_the_assertion_has_teeth_when_live_tokens_are_cut(self, rig: dict[str, Any]) -> None:
        """Sabotage: drop a quarter of the LIVE completion and the two passes must disagree.

        A quarter rather than one token, because the neutrality tolerance cannot resolve one token
        out of 1,536: measured on the L4, cutting a single live token leaves the LoRA gradient at
        cosine 0.99998, above the 0.999 line. Losing exactly one column is the off-by-one the CPU
        tests pin exactly (`test_no_live_token_is_ever_lost`); this check is about the gradient
        comparison itself having teeth at its own tolerance.
        """
        padded = self.padded_inputs(rig, completion_live=self.COMPLETION_LIVE)
        trimmed, _ = trim_micro_batch(padded)
        keep = self.COMPLETION_LIVE - self.COMPLETION_LIVE // 4
        cut = {
            key: (value[:, :keep] if key in COMPLETION_TOKEN_KEYS else value)
            for key, value in trimmed.items()
        }
        loss_full, grad_full = self.lora_gradient(rig, trimmed)
        loss_cut, grad_cut = self.lora_gradient(rig, cut)
        cosine = torch.nn.functional.cosine_similarity(grad_full, grad_cut, dim=0).item()
        logger.info(
            "SABOTAGE cut %d of %d live tokens: cosine=%.6f loss_full=%.6f loss_cut=%.6f",
            self.COMPLETION_LIVE - keep,
            self.COMPLETION_LIVE,
            cosine,
            loss_full,
            loss_cut,
        )
        assert cosine < 0.999
