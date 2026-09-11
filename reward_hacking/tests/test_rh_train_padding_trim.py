"""The padding trim on this arm's trainer: the composed class on CPU, and neutrality on its loss path.

`reward_hacking.train._build_trainer` constructs `games.train.PaddingTrimmedGRPOTrainer` outright. That
class carries two behaviours this arm needs and each is written once, there: the per-micro-batch
padding trim at TRL's `_prepare_inputs` seam (`trim_micro_batch`), and the `log` override that refuses
to write an empty completions buffer over a step whose parquet already holds rows. The slicing itself
is pinned on fake batches by `games/tests/test_games_padding_trim.py`; the guard's own cases and the
negative control that watches TRL's bare `log` write the zero-row file are `test_rh_train_trace_files`.
What this module pins is the composition and the estimator:

* on CPU, one bare instance of the class this arm constructs trims a training micro-batch, logs the
  step's padded and trimmed token totals into the metrics `build_callbacks` copies into `mem_log.csv`,
  and then still refuses an empty-trace overwrite -- both behaviours on the same object;
* this arm's default estimator is one whose normaliser is pad-invariant, which is the premise of the
  neutrality argument (the games commit's argument covered the loss types it trains under; this arm
  trains under `dr_grpo`, whose Liger normaliser divides by rows times `max_completion_length`, a
  config constant, and TRL's own loss does the same);
* on the GPU, the padded and the trimmed pass of one REAL reward-hacking corpus row -- an ILCB problem
  with its grader inlined, templated exactly as `_prepare_run` templates it -- through the production
  loss path (`compute_liger_loss`, LoRA on the discovered targets, Liger fused GRPO loss with
  `dr_grpo`, gradient checkpointing, `Qwen/Qwen3.5-4B`, the trainer built by `_build_trainer` itself)
  give the same loss to `LOSS_REL_TOLERANCE` (sharper than one live token) and a LoRA gradient within
  the padded pipeline's OWN pad-shape dependence measured on the same run (the bound
  `TestNeutralityOnThisArmsLossPath` explains), and both assertions have teeth: cutting a quarter of
  the LIVE completion tokens breaks the gradient bound, cutting a single live token breaks the loss
  check. Opt-in via GAMES_GPU_TESTS=1, the convention the games suite uses, because the only local
  card is shared and every GPU job here goes through `scripts/gpu_preflight.py` and the resource
  limiter.

The GPU passes call `trim_micro_batch` and `compute_liger_loss` directly rather than driving
`_prepare_inputs`: the seam, its training-mode gate and its eval-mode pass-through are pinned on bare
instances by `games/tests/test_games_padding_trim.py`, and the CPU class above pins them on this arm's
own binding of the class. What the GPU class measures is the loss path, on the tensors the seam hands it.

Measured on the L4 (2026-09-03, `Qwen/Qwen3.5-4B`, bf16, the first misspecified-arm corpus row, 1,978
prompt tokens): 6,138 padded columns to 3,514 trimmed; trim deviation 8.3e-3 (cosine 0.9917, norm ratio
0.9941) against the padded pipeline's own pad-shape dependence of 6.8e-3, 8.4e-3 and 8.5e-3 (padded-4096
vs padded-4160, vs padded-2048, vs no prompt pad), bound 1.71e-2; the quarter-block sabotage reads
3.0e-2 (cosine 0.9702), norm ratio 0.786, loss down by exactly a quarter; the one-token sabotage moves
the loss by 6.5e-4 relative while its gradient cosine stays at 0.99998. The wall clock of the passes,
11.8 s padded to 5.0 s trimmed, is the point of the trim. `TestNeutralityOnThisArmsLossPath` explains
why the gradient bound is a same-run control rather than the games test's constant.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from test_rh_train_config import make_config, make_plan
from test_rh_train_trace_files import write_trace
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

import games.train as gt
from games import preflight
from games.train import COMPLETION_TOKEN_KEYS, trim_micro_batch
from grpo.estimator_defaults import GRPO_LOSS_TYPES
from grpo.throughput import discover_lora_targets
from reward_hacking import train as rh_train
from reward_hacking.train_dataset import TRAINING_GRADER_EXPOSURE, build_dataset, resolve_arm_rows
from reward_hacking.train_partition import load_partition

logger = logging.getLogger(__name__)

# The loss types whose normaliser does not see the pad columns: `grpo` averages each row over its
# live tokens, `bnpo` and `dapo` divide by the loss mask's sum, `dr_grpo` by rows times the configured
# `max_completion_length` (Liger `grpo_loss.py` and TRL `_compute_loss`, 1.10.0, read 2026-09-03).
# Liger's `luspo` divides by a pad-inclusive count and would not be neutral, which is why the config
# refuses it under Liger outright (`test_rh_train_config.TestEstimatorPin`); `cispo`, `sapo` and `vespo`
# are not in use here and are left unclassified rather than asserted about.
PAD_INVARIANT_LOSS_TYPES: frozenset[str] = frozenset({"grpo", "bnpo", "dr_grpo", "dapo"})


def fake_batch(
    *,
    prompt_live: int = 5,
    prompt_left_pad: int = 3,
    completion_live: int = 4,
    completion_right_pad: int = 6,
) -> dict[str, Any]:
    """A one-row TRL-shaped micro-batch: left-padded prompt, right-padded completion."""
    prompt_width = prompt_left_pad + prompt_live
    completion_width = completion_live + completion_right_pad
    prompt_mask = torch.zeros(1, prompt_width, dtype=torch.long)
    prompt_mask[:, prompt_left_pad:] = 1
    completion_mask = torch.zeros(1, completion_width, dtype=torch.long)
    completion_mask[:, :completion_live] = 1
    return {
        "prompt_ids": torch.arange(1, prompt_width + 1)[None] * prompt_mask,
        "prompt_mask": prompt_mask,
        "completion_ids": torch.arange(101, completion_width + 101)[None] * completion_mask,
        "completion_mask": completion_mask,
        "advantages": torch.full((1,), 0.7),
        "num_items_in_batch": torch.tensor(64),
    }


class TestTheComposedTrainerOnCpu:
    """One bare instance of the class `_build_trainer` constructs, carrying what both overrides read.

    `super()._prepare_inputs` is stubbed to hand back a fake batch and `GRPOTrainer.log` to record
    whether TRL's write was reached; nothing loads weights. The instance is the reward-hacking
    trainer's own binding of the class, so a subclass quietly reintroduced here would be what these
    tests exercise -- and `test_rh_train_config` pins that none is.
    """

    ROWS_PER_STEP = 8
    STEP = 70

    def bare_trainer(
        self,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        batch: dict[str, Any],
        *,
        buffered: int,
    ) -> Any:
        trainer = rh_train.PaddingTrimmedGRPOTrainer.__new__(rh_train.PaddingTrimmedGRPOTrainer)
        trainer.model = SimpleNamespace(training=True)  # pyright: ignore[reportAttributeAccessIssue]
        trainer._step = 0  # pyright: ignore[reportAttributeAccessIssue]
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}  # pyright: ignore[reportAttributeAccessIssue]
        trainer.accelerator = SimpleNamespace(is_main_process=True)  # pyright: ignore[reportAttributeAccessIssue]
        trainer.log_completions = True  # pyright: ignore[reportAttributeAccessIssue]
        trainer._logs = {"prompt": deque(f"p{i}" for i in range(buffered))}  # pyright: ignore[reportAttributeAccessIssue]
        trainer.args = SimpleNamespace(output_dir=str(output_dir), steps_per_generation=2)  # pyright: ignore[reportAttributeAccessIssue]
        trainer.state = SimpleNamespace(global_step=self.STEP)  # pyright: ignore[reportAttributeAccessIssue]
        monkeypatch.setattr(GRPOTrainer, "_prepare_inputs", lambda _self, _batch: batch)
        return trainer

    def stub_trl_log(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, float]]:
        reached: list[dict[str, float]] = []
        monkeypatch.setattr(
            GRPOTrainer, "log", lambda _self, logs, _start_time=None: reached.append(logs)
        )
        return reached

    def test_a_training_micro_batch_is_trimmed_and_the_step_totals_reach_the_mem_log_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        trainer = self.bare_trainer(
            tmp_path, monkeypatch, fake_batch(), buffered=self.ROWS_PER_STEP
        )
        out = trainer._prepare_inputs({"prompt": ["x"]})
        assert out["prompt_ids"].tolist() == [[4, 5, 6, 7, 8]]
        assert out["completion_ids"].tolist() == [[101, 102, 103, 104]]
        assert trainer._metrics["train"]["padding_trim/padded_tokens"] == [18.0]
        assert trainer._metrics["train"]["padding_trim/trimmed_tokens"] == [9.0]
        assert gt.STEP_PADDED_TOKENS_METRIC not in trainer._metrics["train"]
        # The last micro-batch of the generation batch closes the step's totals and logs them.
        trainer._step = 1
        with caplog.at_level(logging.INFO, logger="games.train"):
            trainer._prepare_inputs({"prompt": ["x"]})
        assert "padding trim: step tokens padded=36 trimmed=18 kept_fraction=0.500" in caplog.text
        assert trainer._metrics["train"][gt.STEP_PADDED_TOKENS_METRIC] == [36.0]
        assert trainer._metrics["train"][gt.STEP_TRIMMED_TOKENS_METRIC] == [18.0]
        # `build_callbacks`, which this trainer calls, hands exactly these columns to the memory
        # monitor, so the totals land in this arm's mem_log.csv beside each step's VRAM reading.
        assert set(gt.PADDING_TRIM_STEP_METRICS) <= set(gt.MEM_LOG_EXTRA_COLUMNS)

    def test_the_same_instance_still_refuses_an_empty_trace_overwrite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trim first, then the guard: the composition, not either behaviour alone."""
        reached = self.stub_trl_log(monkeypatch)
        path = write_trace(tmp_path, self.STEP, self.ROWS_PER_STEP)
        trainer = self.bare_trainer(tmp_path, monkeypatch, fake_batch(), buffered=0)
        assert trainer._prepare_inputs({"prompt": ["x"]})["completion_ids"].shape == (1, 4)
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            trainer.log({"train_runtime": 0.0029})
        assert reached == [], "TRL's log ran and would have written the empty parquet"
        assert preflight.parquet_row_count(path) == self.ROWS_PER_STEP

    def test_a_full_buffer_reaches_trl_after_trimming(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reached = self.stub_trl_log(monkeypatch)
        write_trace(tmp_path, self.STEP, self.ROWS_PER_STEP)
        trainer = self.bare_trainer(
            tmp_path, monkeypatch, fake_batch(), buffered=self.ROWS_PER_STEP
        )
        trainer._prepare_inputs({"prompt": ["x"]})
        trainer.log({"reward": 0.5})
        assert reached == [{"reward": 0.5}]


class TestTheTrimIsNeutralUnderThisArmsEstimator:
    """The neutrality argument rests on the normaliser never seeing a pad column."""

    def test_the_default_estimator_normalises_by_a_pad_invariant_count(self) -> None:
        config = make_config()
        assert config.loss_type in PAD_INVARIANT_LOSS_TYPES, config.loss_type

    def test_the_classification_names_only_real_trl_loss_types(self) -> None:
        assert PAD_INVARIANT_LOSS_TYPES.issubset(GRPO_LOSS_TYPES)
        assert "luspo" not in PAD_INVARIANT_LOSS_TYPES


GPU_TESTS_ENV = "GAMES_GPU_TESTS"


@pytest.mark.skipif(
    os.environ.get(GPU_TESTS_ENV) != "1" or not torch.cuda.is_available(),
    reason=f"GPU neutrality check: set {GPU_TESTS_ENV}=1 on a box with a free CUDA card (preflight + limiter)",
)
class TestNeutralityOnThisArmsLossPath:
    """Padded vs trimmed pass of one real reward-hacking corpus row, loss and LoRA gradient compared.

    The trainer is the one `_build_trainer` builds from a `PreparedRun` assembled the way
    `_prepare_run` assembles it (shared tokenizer resolver, the stored partition, `resolve_arm_rows`,
    `build_dataset`, `discover_lora_targets`), minus the vLLM engine and the jail: `use_liger_kernel`,
    `gradient_checkpointing`, `dr_grpo`, `beta=0`, the coding completion budget as
    `max_completion_length`, LoRA r=16 on every discovered target. LoRA dropout is zeroed so the
    passes see the same network. The completion is synthetic (random ids, the loss does not care),
    the prompt is the real row; only token counts are logged, never text.

    **The loss check is exact to the token.** On-policy `dr_grpo` with `beta=0` and no
    `old_per_token_logps` makes every live position's loss exactly `-advantage`
    (`exp(logp - logp.detach()) == 1`), so the loss is `-advantage * live_count` over a constant: a
    pure function of the live-token count, independent of the model's outputs. The padded and trimmed
    losses come out bit-identical; a single lost or gained live token moves the loss by 1/1536
    (6.5e-4 relative), so `LOSS_REL_TOLERANCE` at 1e-5 -- two orders above fp32 summation-order noise,
    two below one token -- makes the loss assertion the off-by-one detector the gradient bound below
    cannot be, and `test_the_loss_check_catches_a_single_lost_live_token` watches it fire.

    **The gradient tolerance is a same-run control, not a constant.** The games test asserts cosine
    > 0.999 and holds at 2B (0.99992 there; 0.99991 for this row and rig on `Qwen/Qwen3.5-2B`). On
    the 4B this arm trains, the same row gives ~0.992 -- and so does changing NOTHING but the pad
    shape: padded-4096 vs padded-4160 (one more 64-column chunk, the same live tokens, no trim
    anywhere) and vs padded-2048 sit at the same distance, while vs padded-4100 is 0.99998 and two
    passes of one shape are 0.99999. The pad shape selects kernel tilings (the fla chunked DeltaNet
    kernel: layer 0's live outputs move by one bf16 ulp, 3.6e-3 relative, when only the right pad
    changes; the boolean-mask SDPA path against `is_causal`), a one-ulp difference at the bottom
    compounds to ~2% relative by layer 31, and the 4B's backward amplifies that into ~1% of gradient
    direction where the 2B's does not. In production the pad shape is whichever completion in the
    batch ran longest, so the untrimmed pipeline already plays that lottery every step; the trim is
    one more draw from it.

    The lottery has to include a draw that moves the LEFT side, because the trim does: it removes the
    prompt pad too, and TRL hands the model `input_ids` and `attention_mask` only, so Qwen3.5
    assigns `arange` positions and every live token's absolute position shifts by the pad removed.
    That is neutral because RoPE is relative (a uniform shift leaves every full-attention score
    unchanged in exact arithmetic; only the bf16 cos/sin tables round differently) and the Gated
    DeltaNet layers carry no positional encoding and zero their pad positions
    (`apply_mask_to_padding_states`), but a bound calibrated on right-pad changes alone would not
    have shown that the position shift adds nothing, so `ALTERNATIVE_PAD_SHAPES` carries a no-left-pad
    draw beside the two completion widths. The bound then reads: the trim's deviation from the padded
    pass may not exceed `PAD_SHAPE_LOTTERY_FACTOR` times the padded pipeline's own worst deviation
    across those draws, plus the same-shape noise floor; and the block sabotage has to clear that
    same bound on the same run, so the bound is shown to catch a lost block on the very run that
    accepts the trim. Measured on the L4 (2026-09-03, `Qwen/Qwen3.5-4B`, bf16, the first
    misspecified-arm row, 1,978 prompt tokens): the `NEUTRALITY`, `SABOTAGE` and `ONE_TOKEN` log lines are quoted
    in the test docstrings.
    """

    COMPLETION_LIVE = 1536
    COMPLETION_PADDED = 4096
    PROMPT_LEFT_PAD = 64
    # Other pad-shape accidents of the same live tokens, no trim anywhere, as (label, prompt left pad,
    # completion width): one more 64-column completion chunk, half the completion width, and no prompt
    # pad at all -- the left side the trim removes as well (class docstring).
    ALTERNATIVE_PAD_SHAPES: tuple[tuple[str, int, int], ...] = (
        ("completion_4160", PROMPT_LEFT_PAD, 4160),
        ("completion_2048", PROMPT_LEFT_PAD, 2048),
        ("prompt_pad_0", 0, COMPLETION_PADDED),
    )
    PAD_SHAPE_LOTTERY_FACTOR = 2.0
    # 1 - cosine between two passes of the same shape: 1.1e-5 (4B) and 1.0e-5 (2B) on the L4.
    SAME_SHAPE_NOISE = 1e-4
    NORM_TOLERANCE = 0.01
    # One live token is 1/1536 = 6.5e-4 of the loss; fp32 summation-order noise is ~1e-7.
    LOSS_REL_TOLERANCE = 1e-5

    @pytest.fixture(scope="class")
    def rig(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        config = make_config(
            smoke=True, output_dir=str(tmp_path_factory.mktemp("rh-padding-trim-neutrality"))
        )
        tokenizer, template_facts = preflight.resolve_tokenizer(
            config.model_id, thinking=config.thinking
        )
        chat_template_kwargs = cast("dict[str, str]", template_facts["chat_template_kwargs"])
        partition = load_partition(Path(config.partition_path))
        rows, _budget, _subset = resolve_arm_rows(
            config.arm,
            partition,
            tokenizer,
            max_prompt_tokens=config.max_prompt_tokens,
            enable_thinking=config.thinking,
            chat_template_kwargs=chat_template_kwargs,
            exposure=TRAINING_GRADER_EXPOSURE,
        )
        dataset = build_dataset(
            rows[:1],
            tokenizer,
            max_prompt_tokens=config.max_prompt_tokens,
            enable_thinking=config.thinking,
            chat_template_kwargs=chat_template_kwargs,
        )
        lora_targets = discover_lora_targets(config.model_id)
        plan = make_plan()
        prepared = rh_train.PreparedRun(
            config=config,
            plan=plan,
            dataset=dataset,
            tokenizer=tokenizer,
            lora_targets=lora_targets,
            derived={
                "prefilled_think": template_facts["prefilled_think"],
                "meta_parameter_count": 0,
            },
            dtype=torch.bfloat16,
            device={},
            resume_checkpoint=None,
        )
        loaded_at = time.perf_counter()
        trainer = rh_train._build_trainer(prepared)
        assert type(trainer) is gt.PaddingTrimmedGRPOTrainer
        grpo_args = cast("GRPOConfig", trainer.args)
        assert grpo_args.loss_type == "dr_grpo"
        assert grpo_args.use_liger_kernel
        model = cast("torch.nn.Module", trainer.model)
        model.train()
        unwrapped = trainer.accelerator.unwrap_model(model)
        unwrapped.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs
        )
        trainer.current_gradient_accumulation_steps = plan.gradient_accumulation_steps
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        # What TRL tokenises: the templated `prompt` column, no special tokens added.
        templated = cast("str", dataset[0]["prompt"])
        prompt_ids = torch.tensor(tokenizer(templated, add_special_tokens=False)["input_ids"])[None]
        logger.info(
            "RIG model=%s prompt_tokens=%d max_completion_length=%d lora_targets=%d "
            "build_seconds=%.1f allocated_gib=%.2f",
            config.model_id,
            prompt_ids.size(1),
            grpo_args.max_completion_length,
            len(cast("list[str]", lora_targets["target_modules"])),
            time.perf_counter() - loaded_at,
            torch.cuda.memory_allocated() / 2**30,
        )
        return {
            "trainer": trainer,
            "unwrapped": unwrapped,
            "tokenizer": tokenizer,
            "prompt_ids": prompt_ids,
        }

    def padded_inputs(
        self,
        rig: dict[str, Any],
        *,
        completion_live: int,
        completion_width: int,
        prompt_left_pad: int,
    ) -> dict[str, torch.Tensor]:
        """One left-padded prompt and one right-padded completion; the live tokens never depend on the shape."""
        device = rig["trainer"].accelerator.device
        pad_id = rig["tokenizer"].pad_token_id
        prompt_ids = rig["prompt_ids"]
        prompt_ids = torch.cat([torch.full((1, prompt_left_pad), pad_id), prompt_ids], dim=1)
        prompt_mask = torch.ones_like(prompt_ids)
        prompt_mask[:, :prompt_left_pad] = 0
        generator = torch.Generator().manual_seed(7)
        completion_ids = torch.randint(1000, 100000, (1, completion_width), generator=generator)
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
    ) -> tuple[float, torch.Tensor, float, float]:
        """Loss, flattened trainable gradient, wall seconds and peak GiB of one production-path pass."""
        trainer, model = rig["trainer"], rig["trainer"].model
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss = trainer.compute_liger_loss(rig["unwrapped"], inputs)
        loss.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        flat = torch.cat(
            [
                p.grad.float().flatten()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
        )
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        return float(loss.detach().float()), flat, seconds, peak_gib

    @staticmethod
    def deviation(reference: torch.Tensor, other: torch.Tensor) -> tuple[float, float]:
        """(1 - cosine, norm ratio) of `other` against `reference`."""
        cosine = torch.nn.functional.cosine_similarity(reference, other, dim=0).item()
        return 1.0 - cosine, (other.norm() / reference.norm()).item()

    @pytest.fixture(scope="class")
    def passes(self, rig: dict[str, Any]) -> dict[str, Any]:
        """The padded pass, the trimmed pass and the alternative-shape padded passes, once per class."""
        padded = self.padded_inputs(
            rig,
            completion_live=self.COMPLETION_LIVE,
            completion_width=self.COMPLETION_PADDED,
            prompt_left_pad=self.PROMPT_LEFT_PAD,
        )
        trimmed, stats = trim_micro_batch(padded)
        assert (
            stats.trimmed_columns
            == padded["prompt_ids"].size(1) - self.PROMPT_LEFT_PAD + self.COMPLETION_LIVE
        )
        out: dict[str, Any] = {"padded": self.lora_gradient(rig, padded), "trimmed_inputs": trimmed}
        out["trimmed"] = self.lora_gradient(rig, trimmed)
        for label, prompt_left_pad, completion_width in self.ALTERNATIVE_PAD_SHAPES:
            alternative = self.padded_inputs(
                rig,
                completion_live=self.COMPLETION_LIVE,
                completion_width=completion_width,
                prompt_left_pad=prompt_left_pad,
            )
            assert torch.equal(
                alternative["completion_ids"][:, : self.COMPLETION_LIVE],
                padded["completion_ids"][:, : self.COMPLETION_LIVE],
            )
            assert torch.equal(
                alternative["prompt_ids"][:, prompt_left_pad:],
                padded["prompt_ids"][:, self.PROMPT_LEFT_PAD :],
            )
            out[f"padded_{label}"] = self.lora_gradient(rig, alternative)
        out["stats"] = stats
        return out

    def pad_shape_lottery(self, passes: dict[str, Any]) -> dict[str, tuple[float, float]]:
        """Per alternative pad shape: (1 - cosine, norm ratio) of its padded pass against the padded pass."""
        loss_padded, grad_padded, _, _ = passes["padded"]
        lottery: dict[str, tuple[float, float]] = {}
        for label, _, _ in self.ALTERNATIVE_PAD_SHAPES:
            loss_alternative, grad_alternative, _, _ = passes[f"padded_{label}"]
            assert loss_alternative == pytest.approx(loss_padded, rel=self.LOSS_REL_TOLERANCE), (
                label
            )
            lottery[label] = self.deviation(grad_padded, grad_alternative)
        return lottery

    def bound(self, lottery: dict[str, tuple[float, float]]) -> float:
        """The largest deviation any change of pad shape produced, scaled, plus the same-shape floor."""
        shape_deviation = max(deviation for deviation, _ in lottery.values())
        return self.PAD_SHAPE_LOTTERY_FACTOR * shape_deviation + self.SAME_SHAPE_NOISE

    def test_the_trimmed_pass_is_within_the_padded_pipelines_own_pad_shape_dependence(
        self, passes: dict[str, Any]
    ) -> None:
        """L4, 4B, 2026-09-03 (`GAMES_GPU_TESTS=1`, preflight + limiter):
        NEUTRALITY loss_padded=-0.00546875 loss_trimmed=-0.00546875 trim_deviation=8.314e-03
        (cosine 0.991686) norm_ratio_trim=0.994099 pad_shape_lottery={'completion_4160':
        (6.828e-03, 0.993196), 'completion_2048': (8.354e-03, 0.994732), 'prompt_pad_0':
        (8.513e-03, 0.994333)} bound=1.713e-02 padded_columns=6138 trimmed_columns=3514
        grad_elements=32464896 seconds_padded=11.80 seconds_trimmed=4.97 peak_gib_padded=14.45
        peak_gib_trimmed=14.06. An earlier run of the same shapes read 7.4e-3 for the trim against
        6.0e-3 and 7.4e-3 for the two completion widths: the draws move between processes (Triton
        autotune is timing-based), which is one more reason the bound is taken on the same run."""
        loss_padded, grad_padded, seconds_padded, peak_padded = passes["padded"]
        loss_trimmed, grad_trimmed, seconds_trimmed, peak_trimmed = passes["trimmed"]
        trim_deviation, trim_norm_ratio = self.deviation(grad_padded, grad_trimmed)
        lottery = self.pad_shape_lottery(passes)
        stats = passes["stats"]
        logger.info(
            "NEUTRALITY loss_padded=%.8f loss_trimmed=%.8f trim_deviation=%.3e (cosine %.6f) "
            "norm_ratio_trim=%.6f pad_shape_lottery=%s bound=%.3e padded_columns=%d trimmed_columns=%d "
            "grad_elements=%d seconds_padded=%.2f seconds_trimmed=%.2f peak_gib_padded=%.2f "
            "peak_gib_trimmed=%.2f",
            loss_padded,
            loss_trimmed,
            trim_deviation,
            1.0 - trim_deviation,
            trim_norm_ratio,
            {label: (f"{dev:.3e}", f"{ratio:.6f}") for label, (dev, ratio) in lottery.items()},
            self.bound(lottery),
            stats.padded_columns,
            stats.trimmed_columns,
            grad_padded.numel(),
            seconds_padded,
            seconds_trimmed,
            peak_padded,
            peak_trimmed,
        )
        assert loss_trimmed == pytest.approx(loss_padded, rel=self.LOSS_REL_TOLERANCE)
        assert abs(trim_norm_ratio - 1.0) < self.NORM_TOLERANCE
        for label, (_, ratio) in lottery.items():
            assert abs(ratio - 1.0) < self.NORM_TOLERANCE, label
        assert trim_deviation <= self.bound(lottery)

    def cut_live_completion(self, passes: dict[str, Any], *, keep: int) -> dict[str, torch.Tensor]:
        """The trimmed inputs with the live completion block cut down to its first `keep` tokens."""
        return {
            key: (value[:, :keep] if key in COMPLETION_TOKEN_KEYS else value)
            for key, value in passes["trimmed_inputs"].items()
        }

    def test_the_gradient_bound_has_teeth_when_a_block_of_live_tokens_is_cut(
        self, rig: dict[str, Any], passes: dict[str, Any]
    ) -> None:
        """Sabotage: drop a quarter of the LIVE completion block and the two passes must disagree.

        Judged three ways. Two signals a lost block leaves that no kernel tiling can: with `dr_grpo`
        the loss is the advantage-weighted live-token count over a constant, so a quarter fewer
        tokens is a quarter less loss, and the gradient is a sum over live positions, so its norm
        falls well outside the 1% band. And the neutrality bound itself: the cut's deviation has to
        exceed the same pad-shape-lottery bound the trim was accepted under, on the same run, or that
        bound would be a number that accepts anything. A block rather than one token, because one
        token out of 1,536 sits inside any gradient tolerance here (0.99998 measured by the games
        test); the single-token case is the loss check's job, next test.
        L4, 4B, 2026-09-03: SABOTAGE cut 384 of 1536 live tokens: cut_deviation=2.976e-02 (cosine
        0.970235) norm_ratio=0.786096 loss_full=-0.00546875 loss_cut=-0.00410156 bound=1.713e-02.
        """
        loss_full, grad_full, _, _ = passes["trimmed"]
        keep = self.COMPLETION_LIVE - self.COMPLETION_LIVE // 4
        loss_cut, grad_cut, _, _ = self.lora_gradient(
            rig, self.cut_live_completion(passes, keep=keep)
        )
        cut_deviation, cut_norm_ratio = self.deviation(grad_full, grad_cut)
        bound = self.bound(self.pad_shape_lottery(passes))
        logger.info(
            "SABOTAGE cut %d of %d live tokens: cut_deviation=%.3e (cosine %.6f) norm_ratio=%.6f "
            "loss_full=%.8f loss_cut=%.8f bound=%.3e",
            self.COMPLETION_LIVE - keep,
            self.COMPLETION_LIVE,
            cut_deviation,
            1.0 - cut_deviation,
            cut_norm_ratio,
            loss_full,
            loss_cut,
            bound,
        )
        assert loss_cut != pytest.approx(loss_full, rel=self.LOSS_REL_TOLERANCE)
        assert abs(cut_norm_ratio - 1.0) >= self.NORM_TOLERANCE
        assert cut_deviation > bound

    def test_the_loss_check_catches_a_single_lost_live_token(
        self, rig: dict[str, Any], passes: dict[str, Any]
    ) -> None:
        """Sabotage: drop ONE live completion token; the loss must move, whatever the gradient does.

        The off-by-one a trim could plausibly commit. The gradient cosine and norm of the one-token
        cut are logged so the record shows they sit inside their tolerances (the games test measured
        0.99998), which is exactly why the loss tolerance is the check that has to carry this case.
        L4, 4B, 2026-09-03: ONE_TOKEN cut 1 of 1536 live tokens: loss_full=-0.00546875
        loss_cut=-0.00546519 relative=6.510e-04 cut_deviation=1.878e-05 (cosine 0.999981)
        norm_ratio=0.999733 -- inside both gradient tolerances, caught by the loss check alone.
        """
        loss_full, grad_full, _, _ = passes["trimmed"]
        keep = self.COMPLETION_LIVE - 1
        loss_cut, grad_cut, _, _ = self.lora_gradient(
            rig, self.cut_live_completion(passes, keep=keep)
        )
        cut_deviation, cut_norm_ratio = self.deviation(grad_full, grad_cut)
        logger.info(
            "ONE_TOKEN cut 1 of %d live tokens: loss_full=%.8f loss_cut=%.8f relative=%.3e "
            "cut_deviation=%.3e (cosine %.6f) norm_ratio=%.6f",
            self.COMPLETION_LIVE,
            loss_full,
            loss_cut,
            abs(loss_cut - loss_full) / abs(loss_full),
            cut_deviation,
            1.0 - cut_deviation,
            cut_norm_ratio,
        )
        assert loss_cut != pytest.approx(loss_full, rel=self.LOSS_REL_TOLERANCE)
