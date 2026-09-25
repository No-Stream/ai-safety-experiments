"""Two trainer instruments that change no default behaviour: an entropy proxy and a chunked pass.

Both live in `games.train.InstrumentedGRPOTrainer`, the class `PaddingTrimmedGRPOTrainer` extends,
and both exist because a measurement was missing rather than because a run was wrong.

*   **The entropy proxy.** TRL asks vLLM for the sampled token's own log-probability on every
    rollout (`logprobs=0`, `grpo_trainer.py:1117`, kept at `:1830`), and at temperature 1.0 the mean
    surprisal of the sampled tokens is an unbiased single-sample estimate of the sampling
    distribution's entropy. Nothing else in the loop logs entropy at all: the Liger loss path
    appends only `kl` and `clip_ratio` (`grpo_trainer.py:2974-2975`), and Liger forbids both the
    entropy bonus and the top-entropy mask, so every entropy-collapse intervention in the 2025-2026
    literature is unusable here for want of the signal. `sampled_surprisal` reads it off the
    log-probabilities TRL already collected, at the cost of no extra forward pass.
*   **The chunked old-log-probability pass.** With the importance-sampling correction ON, TRL
    recovers this model's own per-token log-probabilities for every generation batch
    (`grpo_trainer.py:2635`), and its `_get_per_token_logps_and_entropies` builds the whole
    (positions x vocab) logits tensor for a row at once (`:1524-1534`). One 32,768-token row over
    Qwen3.5's 248,320-token vocabulary in fp32 is 30.31 GiB, which is the figure that pass died at
    on 2026-08-19. `chunked_per_token_logps` applies the head to hidden-state slices instead, so
    the peak is one chunk wide. The exactness splits in two, and the tests split with it: the half
    chunking introduces is bit-for-bit (a log-softmax reduces each position over the vocabulary and
    reads no other position, so pairing and slicing change nothing), while the half it inherits is
    only floating-point (the head's matmul reassociates under a change of row count).
*   **Log-only mode** computes the correction and logs TRL's own ratio statistics while applying a
    weight of exactly one, so the vLLM-versus-trainer mismatch on this hybrid-attention family can
    be measured without moving the gradient.

Everything is CPU: a tiny `nn.Linear` stands in for the language-model head, and the trainer seams
run on bare instances with `super()` stubbed, the pattern `test_games_padding_trim.py` established.
"""

from __future__ import annotations

import inspect
import logging
import math
import re
from collections import defaultdict
from dataclasses import asdict, fields
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
import torch
from liger_kernel.chunked_loss.grpo_loss import get_gamma_weights
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen2Config
from trl.trainer.utils import selective_log_softmax

import games.train as gt
from games.train import (
    IMPORTANCE_SAMPLING_ZERO_FRACTION_METRIC,
    SAMPLED_SURPRISAL_MAX_METRIC,
    SAMPLED_SURPRISAL_MEAN_METRIC,
    SAMPLED_SURPRISAL_MIN_METRIC,
    SAMPLED_SURPRISAL_TOKENS_METRIC,
    GradientHealthCallback,
    chunked_per_token_logps,
    sampled_surprisal,
)
from grpo.estimator_defaults import (
    VESPO_IMPORTANCE_SAMPLING_MODES,
    VLLM_IMPORTANCE_SAMPLING_MODE,
    VLLM_IMPORTANCE_SAMPLING_MODES,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

HIDDEN_SIZE = 16
VOCAB_SIZE = 37


def head_and_inputs(
    *, rows: int = 3, positions: int = 10, dtype: torch.dtype = torch.float32
) -> tuple[torch.nn.Linear, torch.Tensor, torch.Tensor]:
    """A tiny head, hidden states and sampled ids, seeded so every comparison sees one draw."""
    generator = torch.Generator().manual_seed(0)
    head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=True, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=generator).to(dtype))
        head.bias.copy_(torch.randn(VOCAB_SIZE, generator=generator).to(dtype))
    hidden = torch.randn(rows, positions, HIDDEN_SIZE, generator=generator).to(dtype)
    token_ids = torch.randint(0, VOCAB_SIZE, (rows, positions), generator=generator)
    return head, hidden, token_ids


def unchunked_per_token_logps(
    head: torch.nn.Linear, hidden: torch.Tensor, token_ids: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    """TRL's own arithmetic, whole-sequence: head, divide by temperature, gather the log-softmax.

    Written out rather than imported because it is the claim under test: `grpo_trainer.py`
    :1524-1534 computes `logits = outputs.logits`, slices, divides by `self.temperature` and calls
    `selective_log_softmax`, and this is that with the head applied to the whole sequence at once.
    """
    return selective_log_softmax(head(hidden) / temperature, token_ids)


POSITIONS = 10
# Widths 5 and 10 divide the 10 positions; 1, 3, 4 and 11 deliberately do not, because a
# remainder chunk is where an off-by-one in the slicing would show and nowhere else.
CHUNK_WIDTHS = (1, 2, 3, 4, 5, 10, 11, 1000)


def logits_block_lookup(
    block: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    """A stand-in head that returns a precomputed logits slice, plus the hidden states to drive it.

    The hidden state of a position is its own index, so the fake head answers by indexing rather than
    by a matmul. That isolates exactly the half chunking introduces -- which positions get paired
    with which tokens, and whether a log-softmax cares how many positions sit beside it -- from the
    half it inherits, the head's own reassociation under a change of row count.
    """
    positions = block.size(1)
    hidden = torch.arange(positions, dtype=torch.float32).view(1, positions, 1)
    hidden = hidden.expand(block.size(0), positions, 1)

    def head(slice_of_hidden: torch.Tensor) -> torch.Tensor:
        return block[:, slice_of_hidden[0, :, 0].long(), :]

    return head, hidden


class TestTheSlicingItselfIsBitForBit:
    """Given one logits block, slicing the position axis changes no log-probability at all.

    This is the claim chunking rests on: a log-softmax reduces each position over the vocabulary and
    reads no other position. Held here on a head that only indexes, so the assertion is `torch.equal`
    and not a tolerance.

    Sabotage run once and watched red (2026-09-04): shifting the id slice one position later than
    the hidden-state slice inside `chunked_per_token_logps` failed every width in this class and 29
    tests in this file altogether.
    """

    @pytest.mark.parametrize("chunk_tokens", CHUNK_WIDTHS)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_a_chunk_of_any_width_gives_the_same_log_probabilities(
        self, chunk_tokens: int, dtype: torch.dtype
    ) -> None:
        _, _, token_ids = head_and_inputs()
        block = torch.randn(
            3, POSITIONS, VOCAB_SIZE, generator=torch.Generator().manual_seed(7)
        ).to(dtype)
        head, hidden = logits_block_lookup(block)
        chunked = chunked_per_token_logps(
            hidden, head, token_ids, temperature=1.0, chunk_tokens=chunk_tokens
        )
        assert chunked.shape == token_ids.shape
        assert torch.equal(chunked, selective_log_softmax(block, token_ids))

    def test_the_temperature_divides_the_logits_before_the_softmax(self) -> None:
        """A temperature other than one must reach the logits, not the log-probabilities."""
        _, _, token_ids = head_and_inputs()
        block = torch.randn(3, POSITIONS, VOCAB_SIZE, generator=torch.Generator().manual_seed(7))
        head, hidden = logits_block_lookup(block)
        chunked = chunked_per_token_logps(hidden, head, token_ids, temperature=0.7, chunk_tokens=4)
        assert torch.equal(chunked, selective_log_softmax(block / 0.7, token_ids))
        assert not torch.equal(chunked, selective_log_softmax(block, token_ids))

    def test_the_result_is_the_naive_log_softmax_and_gather(self) -> None:
        """Independently of TRL's helper: log_softmax then gather, the definition."""
        _, _, token_ids = head_and_inputs()
        block = torch.randn(3, POSITIONS, VOCAB_SIZE, generator=torch.Generator().manual_seed(7))
        head, hidden = logits_block_lookup(block)
        naive = torch.gather(block.log_softmax(-1), dim=-1, index=token_ids.unsqueeze(-1)).squeeze(
            -1
        )
        chunked = chunked_per_token_logps(hidden, head, token_ids, temperature=1.0, chunk_tokens=4)
        assert torch.allclose(chunked, naive, atol=1e-6)


class TestTheWholePassMatchesTheUnchunkedOneToTheHeadsOwnPrecision:
    """End to end through a real matmul the agreement is floating point, not bit, and that is why
    `cast_lm_head_to_fp32` exists.

    Measured on this box (2026-09-04): a different row count reassociates the head's reduction over
    the hidden dimension, which moves a log-probability by a mean 8e-8 nats in fp32 and a mean 0.0024
    nats (max one bfloat16 unit in the last place) in bfloat16. The fp32 figure is nothing; the
    bfloat16 figure summed over thousands of tokens is not, which is the caveat the log-only
    importance-sampling mode carries and the reason it asks for the fp32 head.
    """

    FP32_TOLERANCE_NATS = 1e-5
    BFLOAT16_MEAN_TOLERANCE_NATS = 0.05

    @pytest.mark.parametrize("chunk_tokens", CHUNK_WIDTHS)
    def test_fp32_agrees_to_a_ten_thousandth_of_a_bit(self, chunk_tokens: int) -> None:
        head, hidden, token_ids = head_and_inputs()
        chunked = chunked_per_token_logps(
            hidden, head, token_ids, temperature=1.0, chunk_tokens=chunk_tokens
        )
        whole = unchunked_per_token_logps(head, hidden, token_ids, temperature=1.0)
        assert torch.allclose(chunked, whole, rtol=0.0, atol=self.FP32_TOLERANCE_NATS)

    def test_bfloat16_agrees_within_the_heads_own_last_bit(self) -> None:
        """The production pass sees bf16 logits, where TRL takes its per-row log-softmax branch."""
        head, hidden, token_ids = head_and_inputs(dtype=torch.bfloat16)
        chunked = chunked_per_token_logps(hidden, head, token_ids, temperature=1.0, chunk_tokens=3)
        whole = unchunked_per_token_logps(head, hidden, token_ids, temperature=1.0)
        deviation = (chunked.detach().float() - whole.detach().float()).abs()
        assert float(deviation.mean()) < self.BFLOAT16_MEAN_TOLERANCE_NATS

    def test_a_chunk_at_or_above_the_sequence_length_is_the_unchunked_pass_exactly(self) -> None:
        """One slice is one matmul of the original shape, so here the agreement IS bit-for-bit."""
        head, hidden, token_ids = head_and_inputs()
        for chunk_tokens in (POSITIONS, POSITIONS + 1, 2048):
            assert torch.equal(
                chunked_per_token_logps(
                    hidden, head, token_ids, temperature=1.0, chunk_tokens=chunk_tokens
                ),
                unchunked_per_token_logps(head, hidden, token_ids, temperature=1.0),
            ), chunk_tokens

    def test_a_single_position_sequence_is_still_one_chunk(self) -> None:
        head, hidden, token_ids = head_and_inputs(positions=1)
        assert torch.equal(
            chunked_per_token_logps(hidden, head, token_ids, temperature=1.0, chunk_tokens=2048),
            unchunked_per_token_logps(head, hidden, token_ids, temperature=1.0),
        )


class TestTheChunkWidthIsChecked:
    @pytest.mark.parametrize("chunk_tokens", [0, -1, -2048])
    def test_a_chunk_below_one_token_is_refused(self, chunk_tokens: int) -> None:
        """Zero would loop forever and a negative would silently drop every position."""
        head, hidden, token_ids = head_and_inputs()
        with pytest.raises(ValueError, match="at least one token"):
            chunked_per_token_logps(
                hidden, head, token_ids, temperature=1.0, chunk_tokens=chunk_tokens
            )

    def test_hidden_states_and_ids_of_different_widths_are_refused(self) -> None:
        """The one misalignment the chunking could hide: a slice pairing off-by-one columns."""
        head, hidden, token_ids = head_and_inputs()
        with pytest.raises(RuntimeError, match="positions"):
            chunked_per_token_logps(
                hidden, head, token_ids[:, :-1], temperature=1.0, chunk_tokens=4
            )


class TestTheChunkedPassNeverBuildsTheWholeVocabularyBlock:
    """The point of the exercise: the widest logits tensor the pass allocates is one chunk."""

    def test_the_head_is_only_ever_asked_for_a_chunk_of_positions(self) -> None:
        head, hidden, token_ids = head_and_inputs(positions=10)
        widths: list[int] = []

        def recording_head(slice_of_hidden: torch.Tensor) -> torch.Tensor:
            widths.append(slice_of_hidden.size(1))
            return head(slice_of_hidden)

        chunked_per_token_logps(hidden, recording_head, token_ids, temperature=1.0, chunk_tokens=4)
        assert widths == [4, 4, 2]
        assert max(widths) == 4


class TestSampledSurprisal:
    """Minus the sampled log-probability, averaged over live tokens only.

    Sabotage run once and watched red (2026-09-04): dropping the minus sign in `sampled_surprisal`
    took all five tests here plus three of the seam's down, every reading having turned negative.
    """

    LOGPS = torch.tensor([[-0.5, -1.5, -2.5, 0.0], [-1.0, -3.0, 0.0, 0.0]])
    MASK = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    def test_the_mean_is_minus_the_mean_sampled_logprob_over_live_tokens(self) -> None:
        reading = sampled_surprisal(self.LOGPS, self.MASK)
        assert reading is not None
        assert reading.token_weighted_mean == pytest.approx((0.5 + 1.5 + 2.5 + 1.0 + 3.0) / 5)
        assert reading.live_tokens == 5
        assert reading.completions == 2

    def test_the_extremes_are_per_completion_means_not_per_token(self) -> None:
        reading = sampled_surprisal(self.LOGPS, self.MASK)
        assert reading is not None
        assert reading.min_completion_mean == pytest.approx((0.5 + 1.5 + 2.5) / 3)
        assert reading.max_completion_mean == pytest.approx((1.0 + 3.0) / 2)

    def test_the_padding_columns_do_not_pull_the_mean_toward_zero(self) -> None:
        """A pad's log-probability is stored as 0.0, which reads as certainty if it is counted."""
        reading = sampled_surprisal(self.LOGPS, self.MASK)
        assert reading is not None
        counting_the_pads = float(-self.LOGPS.mean())
        assert reading.token_weighted_mean > counting_the_pads
        assert counting_the_pads == pytest.approx((0.5 + 1.5 + 2.5 + 1.0 + 3.0) / 8)

    def test_a_completion_with_no_live_token_is_left_out_of_the_extremes(self) -> None:
        logps = torch.tensor([[-2.0, -2.0], [0.0, 0.0]])
        mask = torch.tensor([[1, 1], [0, 0]])
        reading = sampled_surprisal(logps, mask)
        assert reading is not None
        assert reading.completions == 1
        assert reading.min_completion_mean == pytest.approx(2.0)
        assert reading.max_completion_mean == pytest.approx(2.0)

    def test_a_batch_with_no_live_token_at_all_reads_absent_not_zero(self) -> None:
        """Zero surprisal is a real claim (a deterministic policy); no denominator is not."""
        assert sampled_surprisal(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.long)) is None

    def test_a_uniform_policy_reads_the_entropy_of_the_uniform_distribution(self) -> None:
        """The estimator's meaning, checked against a distribution whose entropy is known."""
        vocab = 8
        logps = torch.full((4, 6), -math.log(vocab))
        reading = sampled_surprisal(logps, torch.ones(4, 6, dtype=torch.long))
        assert reading is not None
        assert reading.token_weighted_mean == pytest.approx(math.log(vocab))


class TestGradientHealthGuard:
    def test_a_zero_gradient_with_reward_variance_fails_at_the_logged_step(self) -> None:
        callback = GradientHealthCallback()
        with pytest.raises(RuntimeError, match=r"dead gradient.*step 3"):
            callback.on_log(
                SimpleNamespace(),
                SimpleNamespace(global_step=3),
                SimpleNamespace(),
                logs={"learning_rate": 0.0, "grad_norm": 0.0, "frac_reward_zero_std": 0.25},
            )

    def test_a_nonzero_first_step_gradient_survives_zero_warmup_learning_rate(self) -> None:
        callback = GradientHealthCallback()
        callback.on_log(
            SimpleNamespace(),
            SimpleNamespace(global_step=1),
            SimpleNamespace(),
            logs={"learning_rate": 0.0, "grad_norm": 0.1, "frac_reward_zero_std": 0.25},
        )

    def test_an_all_pure_step_is_exempt_because_no_gradient_is_expected(self) -> None:
        callback = GradientHealthCallback()
        callback.on_log(
            SimpleNamespace(),
            SimpleNamespace(global_step=2),
            SimpleNamespace(),
            logs={"grad_norm": 0.0, "frac_reward_zero_std": 1.0},
        )


class TestImportanceSamplingMaskFraction:
    def test_sequence_zero_fraction_is_recorded_before_log_only_neutralises_the_ratio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        batch = generation_batch()
        batch["importance_sampling_ratio"] = torch.tensor([[0.0], [0.25]])
        batch["old_per_token_logps"] = torch.zeros_like(
            batch["completion_ids"], dtype=torch.float32
        )
        trainer = bare_trainer(monkeypatch, batch, log_only=True)

        result = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]

        assert trainer._metrics["train"][IMPORTANCE_SAMPLING_ZERO_FRACTION_METRIC] == [0.5]  # pyright: ignore[reportPrivateUsage]
        assert torch.equal(result["importance_sampling_ratio"], torch.ones(2, 1))


def generation_batch(*, with_sampled_logprobs: bool = True) -> dict[str, Any]:
    """A TRL-shaped generation batch: the completion mask plus, optionally, vLLM's log-probabilities."""
    batch: dict[str, Any] = {
        "prompt_ids": torch.arange(1, 7).repeat(2, 1),
        "prompt_mask": torch.ones(2, 6, dtype=torch.long),
        "completion_ids": torch.arange(11, 15).repeat(2, 1),
        "completion_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "advantages": torch.tensor([0.5, -0.5]),
    }
    if with_sampled_logprobs:
        batch["sampling_per_token_logps"] = torch.tensor(
            [[-0.5, -1.5, -2.5, 0.0], [-1.0, -3.0, 0.0, 0.0]]
        )
    return batch


TRL_RATIO_METRIC = "sampling/importance_sampling_ratio/mean"


def bare_trainer(
    monkeypatch: pytest.MonkeyPatch,
    batch: dict[str, Any],
    *,
    training: bool = True,
    log_only: bool = False,
    gradient_accumulation_steps: int = 2,
) -> gt.PaddingTrimmedGRPOTrainer:
    """The seam on a bare instance, with `GRPOTrainer._generate_and_score_completions` stubbed.

    The stub appends the ratio metric TRL appends itself (`grpo_trainer.py:2875-2882`), so a test
    can tell "the statistics were logged" from "the weight was neutralised".
    """
    trainer = gt.PaddingTrimmedGRPOTrainer.__new__(gt.PaddingTrimmedGRPOTrainer)
    trainer.model = SimpleNamespace(training=training)  # pyright: ignore[reportAttributeAccessIssue]
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}  # pyright: ignore[reportAttributeAccessIssue]
    trainer.num_iterations = 1  # pyright: ignore[reportAttributeAccessIssue]
    trainer.args = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        steps_per_generation=2, gradient_accumulation_steps=gradient_accumulation_steps
    )
    trainer.importance_sampling_log_only = log_only

    def stub(bound: gt.PaddingTrimmedGRPOTrainer, _batch: object) -> dict[str, Any]:
        bound._metrics["train"][TRL_RATIO_METRIC].append(1.02)  # pyright: ignore[reportPrivateUsage]
        return batch

    monkeypatch.setattr(gt.GRPOTrainer, "_generate_and_score_completions", stub)
    return trainer


class TestTheEntropyProxySeam:
    def test_a_generation_batch_records_the_four_readings_and_says_them_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        trainer = bare_trainer(monkeypatch, generation_batch())
        with caplog.at_level("INFO", logger="games.train"):
            trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        metrics = trainer._metrics["train"]  # pyright: ignore[reportPrivateUsage]
        assert metrics[SAMPLED_SURPRISAL_MEAN_METRIC] == [pytest.approx(1.7)]
        assert metrics[SAMPLED_SURPRISAL_MIN_METRIC] == [pytest.approx(1.5)]
        assert metrics[SAMPLED_SURPRISAL_MAX_METRIC] == [pytest.approx(2.0)]
        assert metrics[SAMPLED_SURPRISAL_TOKENS_METRIC] == [5.0]
        assert "sampled surprisal" in caplog.text

    def test_every_reading_reaches_the_memory_log_beside_the_phase_seconds(self) -> None:
        assert set(gt.SAMPLED_SURPRISAL_METRICS) <= set(gt.MEM_LOG_EXTRA_COLUMNS)
        assert gt.SAMPLED_SURPRISAL_METRICS == (
            SAMPLED_SURPRISAL_MEAN_METRIC,
            SAMPLED_SURPRISAL_MIN_METRIC,
            SAMPLED_SURPRISAL_MAX_METRIC,
            SAMPLED_SURPRISAL_TOKENS_METRIC,
        )

    def test_the_train_summary_watches_the_proxy_without_requiring_it(self) -> None:
        """Optional, so a rollout path with no log-probabilities does not fail the read-back; watched,
        so the constant-metric warning covers an instrument that stopped moving."""
        assert set(gt.SAMPLED_SURPRISAL_METRICS) <= set(gt.OPTIONAL_METRICS)
        assert not set(gt.SAMPLED_SURPRISAL_METRICS) & set(gt.REQUIRED_METRICS)

    def test_a_recorded_run_history_lands_in_the_read_back_summary(self) -> None:
        """The whole way through: two steps of log records, and the summary carries the trajectory."""
        history = [
            {"reward": 0.4, SAMPLED_SURPRISAL_MEAN_METRIC: 1.8},
            {"reward": 0.5, SAMPLED_SURPRISAL_MEAN_METRIC: 1.4},
        ]
        trainer = SimpleNamespace(state=SimpleNamespace(log_history=history))
        summary, _ = gt.read_back_metrics(
            cast("gt.GRPOTrainer", trainer),
            required=("reward",),
            constant_by_construction=gt.constant_by_construction_metrics(
                gt.PARSE_PENALTY_MARGIN_BELOW_WORSE
            ),
        )
        flat = SAMPLED_SURPRISAL_MEAN_METRIC.replace("/", "_")
        assert summary[f"{flat}_final"] == pytest.approx(1.4)
        assert summary[f"{flat}_mean"] == pytest.approx(1.6)

    def test_a_rollout_path_that_returns_no_logprobs_leaves_the_metric_absent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The transformers.generate path sets `logprobs = None` (`grpo_trainer.py:1918`). Absent,
        never zero: a zero would read as a collapsed policy."""
        trainer = bare_trainer(monkeypatch, generation_batch(with_sampled_logprobs=False))
        with caplog.at_level("INFO", logger="games.train"):
            trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
            trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        metrics = trainer._metrics["train"]  # pyright: ignore[reportPrivateUsage]
        for name in gt.SAMPLED_SURPRISAL_METRICS:
            assert name not in metrics, name
        assert caplog.text.count("no sampled log-probabilities") == 1

    def test_an_eval_batch_records_under_the_eval_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = bare_trainer(monkeypatch, generation_batch(), training=False)
        trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        for name in gt.SAMPLED_SURPRISAL_METRICS:
            assert name not in trainer._metrics["train"], name  # pyright: ignore[reportPrivateUsage]
        assert trainer._metrics["eval"][SAMPLED_SURPRISAL_MEAN_METRIC] == [pytest.approx(1.7)]  # pyright: ignore[reportPrivateUsage]

    def test_a_tool_mask_narrows_the_live_tokens_the_way_trl_narrows_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TRL's own mismatch statistics multiply the two masks (`grpo_trainer.py:2851`)."""
        batch = generation_batch()
        batch["tool_mask"] = torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1]])
        trainer = bare_trainer(monkeypatch, batch)
        trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        metrics = trainer._metrics["train"]  # pyright: ignore[reportPrivateUsage]
        assert metrics[SAMPLED_SURPRISAL_TOKENS_METRIC] == [4.0]
        assert metrics[SAMPLED_SURPRISAL_MEAN_METRIC] == [
            pytest.approx((0.5 + 2.5 + 1.0 + 3.0) / 4)
        ]


class TestLogOnlyImportanceSampling:
    """Compute the correction, log its statistics, weight the gradient by exactly one.

    Two sabotages run once and watched red (2026-09-04): leaving `old_per_token_logps` in the batch
    moved the log-only loss to -0.0687 against correction-off's -0.0625, and leaving the ratio as
    TRL computed it moved it too. Both were caught by
    `test_log_only_leaves_the_loss_identical_to_correction_off`, so the loss comparison rather than
    the key-level assertions is what carries the claim.
    """

    def correction_on_batch(self, *, mode: str = VLLM_IMPORTANCE_SAMPLING_MODE) -> dict[str, Any]:
        """What TRL hands back with the correction ON: a ratio, and recomputed old log-probabilities.

        The ratio's shape follows the mode, as TRL's own does (`grpo_trainer.py:2653-2664`): one
        column per sequence under the sequence modes, one per completion token under the token modes.
        """
        batch = generation_batch()
        batch["importance_sampling_ratio"] = (
            torch.tensor([[0.94, 1.11, 0.87, 1.0], [1.07, 0.92, 1.0, 1.0]])
            if mode.startswith("token_")
            else torch.tensor([[0.94], [1.07]])
        )
        batch["old_per_token_logps"] = torch.tensor(
            [[-0.52, -1.44, -2.61, 0.0], [-1.03, -2.88, 0.0, 0.0]]
        )
        return batch

    def test_the_ratio_becomes_exactly_one_and_keeps_its_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = bare_trainer(monkeypatch, self.correction_on_batch(), log_only=True)
        out = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert torch.equal(out["importance_sampling_ratio"], torch.ones(2, 1))

    def test_the_statistics_trl_logged_are_left_in_place(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: the mismatch is measured, the gradient is not moved."""
        trainer = bare_trainer(monkeypatch, self.correction_on_batch(), log_only=True)
        trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert trainer._metrics["train"][TRL_RATIO_METRIC] == [1.02]  # pyright: ignore[reportPrivateUsage]

    def test_the_recomputed_old_logprobs_are_dropped_when_trl_only_had_them_for_the_correction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Aligned generation and optimizer steps: correction-off would have passed None here, so
        keeping the recomputed values would leave the surrogate ratio near one instead of at one."""
        trainer = bare_trainer(
            monkeypatch, self.correction_on_batch(), log_only=True, gradient_accumulation_steps=2
        )
        out = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert "old_per_token_logps" not in out

    def test_the_old_logprobs_are_kept_when_the_steps_are_misaligned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Misaligned, TRL needs them for its own off-policy correction, which log-only must not
        touch: dropping them there would change the estimator rather than leave it alone."""
        trainer = bare_trainer(
            monkeypatch, self.correction_on_batch(), log_only=True, gradient_accumulation_steps=3
        )
        out = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert "old_per_token_logps" in out

    def test_correction_on_without_log_only_is_left_exactly_as_trl_built_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        batch = self.correction_on_batch()
        expected = batch["importance_sampling_ratio"].clone()
        trainer = bare_trainer(monkeypatch, batch, log_only=False)
        out = trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]
        assert torch.equal(out["importance_sampling_ratio"], expected)
        assert "old_per_token_logps" in out


def dr_grpo_loss(inputs: dict[str, Any], *, correction: bool) -> float:
    """Run TRL 1.10's own `_compute_loss` body over one synthetic micro-batch, on CPU.

    A bare `GRPOTrainer` with the attributes that path reads, and the model forward replaced by a
    fixed log-probability tensor, so the only thing varying between calls is what `inputs` carries.
    That is what makes "log-only leaves the loss alone" a claim about TRL's arithmetic rather than
    about a re-implementation of it.
    """
    trainer = gt.GRPOTrainer.__new__(gt.GRPOTrainer)
    per_token_logps = torch.tensor([[-0.51, -1.47, -2.55, 0.0], [-1.02, -2.95, 0.0, 0.0]])
    entropies = torch.full_like(per_token_logps, 0.3)
    trainer._get_per_token_logps_and_entropies = (  # pyright: ignore[reportAttributeAccessIssue]
        lambda *_args, **_kwargs: (per_token_logps, entropies, None)
    )
    trainer.model = SimpleNamespace(training=True)  # pyright: ignore[reportAttributeAccessIssue]
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}  # pyright: ignore[reportAttributeAccessIssue]
    trainer.accelerator = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        reduce=lambda tensor, reduction: tensor,
        gather=lambda tensor: tensor,
        sync_gradients=True,
    )
    trainer.top_entropy_quantile = 1.0
    trainer.off_policy_mask_threshold = None
    trainer.importance_sampling_level = "token"
    trainer.beta = 0.0
    trainer.loss_type = "dr_grpo"
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    trainer.max_completion_length = 4
    trainer.current_gradient_accumulation_steps = 1
    trainer.use_vllm = True
    trainer.vllm_importance_sampling_correction = correction
    trainer.aux_loss_enabled = False
    trainer._entropy_bonus_enabled = False  # pyright: ignore[reportAttributeAccessIssue]
    trainer.args = SimpleNamespace(delta=None)  # pyright: ignore[reportAttributeAccessIssue]
    return float(trainer._compute_loss(None, inputs))  # pyright: ignore[reportPrivateUsage]


class TestLogOnlyLeavesTheGradientWhereCorrectionOffLeavesIt:
    """The claim that makes log-only a measurement rather than a treatment, on TRL's own loss."""

    def log_only_inputs(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        batch = TestLogOnlyImportanceSampling().correction_on_batch()
        trainer = bare_trainer(monkeypatch, batch, log_only=True)
        return trainer._generate_and_score_completions({"prompt": ["x"]})  # pyright: ignore[reportPrivateUsage]

    def test_log_only_leaves_the_loss_identical_to_correction_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        off = dr_grpo_loss(generation_batch(), correction=False)
        log_only = dr_grpo_loss(self.log_only_inputs(monkeypatch), correction=True)
        assert log_only == off

    def test_the_uncorrected_ratio_would_have_moved_it_so_the_check_has_teeth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del monkeypatch
        off = dr_grpo_loss(generation_batch(), correction=False)
        weighted = dr_grpo_loss(
            TestLogOnlyImportanceSampling().correction_on_batch(), correction=True
        )
        assert weighted != off

    @pytest.mark.parametrize("mode", VLLM_IMPORTANCE_SAMPLING_MODES)
    def test_log_only_neutralises_the_ratio_of_every_mode(
        self, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """The four modes hand the loss two different SHAPES, and log-only has to neutralise both.

        The sequence modes give one ratio per sequence, (B, 1), broadcast over that sequence's tokens;
        the token modes give (B, T). Log-only is `torch.ones_like`, which is shape-agnostic, and this
        is what pins that: a mode-specific neutralisation, or one that assumed (B, 1), would leave a
        token-level ratio weighting the gradient while `run_config.json` recorded log-only.
        """
        batch = TestLogOnlyImportanceSampling().correction_on_batch(mode=mode)
        expected_shape = batch["importance_sampling_ratio"].shape
        neutralised = bare_trainer(
            monkeypatch, batch, log_only=True
        )._generate_and_score_completions(  # pyright: ignore[reportPrivateUsage]
            {"prompt": ["x"]}
        )
        assert neutralised["importance_sampling_ratio"].shape == expected_shape
        assert torch.equal(neutralised["importance_sampling_ratio"], torch.ones(expected_shape))
        assert dr_grpo_loss(neutralised, correction=True) == dr_grpo_loss(
            generation_batch(), correction=False
        )

    @pytest.mark.parametrize("mode", VLLM_IMPORTANCE_SAMPLING_MODES)
    def test_each_modes_own_ratio_would_have_moved_the_loss(self, mode: str) -> None:
        """Per mode, so the parametrised neutrality check above cannot pass on a ratio of all ones."""
        off = dr_grpo_loss(generation_batch(), correction=False)
        weighted = dr_grpo_loss(
            TestLogOnlyImportanceSampling().correction_on_batch(mode=mode), correction=True
        )
        assert weighted != off


def make_config(**overrides: object) -> gt.GameTrainConfig:
    """A valid config, with whatever the test cares about overridden."""
    base: dict[str, object] = {"arm": "twin-pd-group", "generate_fresh": True}
    return gt.GameTrainConfig(**(base | overrides))  # pyright: ignore[reportArgumentType]


class TestTheKnobsDefaultToTodaysBehaviour:
    """A new flag whose default changes a run is a silent relabelling of every arm that follows."""

    def test_the_chunk_default_is_the_documented_two_thousand_and_forty_eight(self) -> None:
        assert gt.OLD_LOGPS_CHUNK_TOKENS == 2048

    def test_a_config_defaults_to_no_log_only_and_the_default_chunk(self) -> None:
        defaults = {field.name: field.default for field in fields(gt.GameTrainConfig)}
        assert defaults["vllm_importance_sampling_log_only"] is False
        assert defaults["old_logps_chunk_tokens"] == gt.OLD_LOGPS_CHUNK_TOKENS

    def test_both_knobs_land_in_the_run_record(self) -> None:
        recorded = asdict(make_config())
        assert recorded["old_logps_chunk_tokens"] == gt.OLD_LOGPS_CHUNK_TOKENS
        assert recorded["vllm_importance_sampling_log_only"] is False

    def test_the_command_line_defaults_match_the_dataclass_defaults(self) -> None:
        config = gt._parse_args(["--arm", "twin-pd-group", "--generate-fresh"])  # pyright: ignore[reportPrivateUsage]
        assert config.old_logps_chunk_tokens == gt.OLD_LOGPS_CHUNK_TOKENS
        assert config.vllm_importance_sampling_log_only is False

    def test_the_flags_are_spelled_the_way_the_kit_will_spell_them(self) -> None:
        config = gt._parse_args(  # pyright: ignore[reportPrivateUsage]
            [
                "--arm",
                "twin-pd-group",
                "--generate-fresh",
                "--old-logps-chunk-tokens",
                "512",
                "--vllm-importance-sampling-log-only",
            ]
        )
        assert config.old_logps_chunk_tokens == 512
        assert config.vllm_importance_sampling_log_only is True


class TestTheConfigRefusesAnImpossibleInstrument:
    def test_a_chunk_below_one_token_is_refused_at_startup(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            make_config(old_logps_chunk_tokens=0)

    def test_log_only_without_the_correction_is_refused(self) -> None:
        """There is no ratio to log with the correction off, so the mode would measure nothing."""
        with pytest.raises(ValueError, match="log-only"):
            make_config(
                vllm_importance_sampling_log_only=True,
                vllm_importance_sampling_correction=False,
            )

    def test_vespo_with_the_correction_on_is_refused_because_trl_refuses_it(self) -> None:
        """TRL 1.10 requires `vllm_importance_sampling_mode` in {token_truncate, token_mask} for
        vespo with the correction ON (`grpo_trainer.py:923-928`), and nothing here moves the mode off
        TRL's `sequence_mask` default, so the pair cannot run at all. Refused at config time rather
        than inside `GRPOTrainer.__init__`, where TRL's own raise lands after a card is reserved.

        NOT refused for the reason an earlier message gave -- that a ratio of one is not a no-op under
        vespo. It is one: `get_gamma_weights` adds `clamp(log(ratio))` to the sequence log-ratio in
        both implementations (TRL `grpo_trainer.py:3042-3045`, liger `grpo_loss.py:36-38`), and
        log(1) is exactly 0.

        Sabotage run once and watched red (2026-09-04): narrowing the refusal back to log-only-plus-
        vespo failed this test.
        """
        with pytest.raises(ValueError, match="vespo"):
            make_config(loss_type="vespo", acknowledge_liger_estimator_mismatch=True)

    def test_log_only_under_vespo_is_refused_by_the_same_rule(self) -> None:
        """Log-only needs the correction ON, so it is inside the refusal above rather than beside it."""
        with pytest.raises(ValueError, match="vespo"):
            make_config(
                vllm_importance_sampling_log_only=True,
                loss_type="vespo",
                acknowledge_liger_estimator_mismatch=True,
            )

    def test_vespo_is_measurable_with_the_correction_off(self) -> None:
        """The escape hatch the message names has to actually exist."""
        config = make_config(
            loss_type="vespo",
            acknowledge_liger_estimator_mismatch=True,
            vllm_importance_sampling_correction=False,
        )
        assert config.loss_type == "vespo"

    def test_a_ratio_of_one_really_is_a_no_op_under_vespos_gamma_weight(self) -> None:
        """The claim the corrected message rests on, on Liger's own `get_gamma_weights`.

        Held here rather than argued in a comment, because the previous message asserted the opposite
        and nothing went red. A neutralised ratio adds `sum(clamp(log(1)))` to the sequence log-ratio,
        which is zero, so the weight is bit-identical to the correction-off one.
        """
        advantages = torch.tensor([0.5, -0.5])
        log_ratio = torch.zeros(2, 4)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        without = get_gamma_weights(advantages, log_ratio, mask, None)
        neutralised = get_gamma_weights(advantages, log_ratio, mask, torch.ones(2, 1))
        weighted = get_gamma_weights(advantages, log_ratio, mask, torch.tensor([[0.94], [1.07]]))
        assert torch.equal(neutralised, without)
        assert not torch.equal(weighted, without)

    def test_log_only_with_the_correction_on_is_allowed(self) -> None:
        config = make_config(vllm_importance_sampling_log_only=True)
        assert config.vllm_importance_sampling_correction is True

    @pytest.mark.parametrize("sampler_override", [{"top_p": 0.95}, {"top_k": 20}])
    @pytest.mark.parametrize("mode", ["sequence_mask", "token_truncate"])
    def test_a_truncating_sampler_under_the_correction_is_refused(
        self, sampler_override: dict[str, float], mode: str
    ) -> None:
        """TRL's colocated engine reports post-truncation log-probs (`processed_logprobs`) while the
        trainer scores the full distribution, so every sampled token looks more likely to vLLM and the
        ratio is biased below one. The 9B probe (2026-09-24, top_p 0.95, ~8K-token completions) put
        every sequence ratio at 1e-13 or below, zeroing the whole gradient with every signal green.
        """
        with pytest.raises(ValueError, match="post-truncation"):
            make_config(vllm_importance_sampling_mode=mode, **sampler_override)

    def test_a_truncating_sampler_is_allowed_when_the_correction_only_logs(self) -> None:
        config = make_config(top_p=0.95, vllm_importance_sampling_log_only=True)
        assert config.top_p == 0.95


class TestTheImportanceSamplingModeIsSelectable:
    """Which of TRL's four correction modes a run uses, at TRL's own default and recorded per run.

    The mode decides what the correction DOES with the log-probability difference. TRL's default
    aggregates it per sequence and zeroes any sequence whose ratio leaves [C_min, C_max], so on long
    completions a rollout is discarded for being long rather than for being off-policy: the 2B probe
    smoke measured sequence-level ratios averaging 0.66 and 0.21 where the per-token difference was
    0.015 nats. The token-level truncated mode clips each token's own ratio instead, and is what the
    owner precommitted (2026-09-04) arm 1 and its control run if the 9B probe confirms the mismatch.
    """

    def test_the_default_is_trls_own_default(self) -> None:
        """Adopted rather than pinned by hand: a TRL version bump that moved it is visible in the
        run record, and every plan written before this knob existed keeps its own treatment."""
        trl_default = {field.name: field.default for field in fields(gt.GRPOConfig)}[
            "vllm_importance_sampling_mode"
        ]
        assert trl_default == VLLM_IMPORTANCE_SAMPLING_MODE
        assert make_config().vllm_importance_sampling_mode == trl_default

    def test_the_four_names_are_the_ones_trls_ratio_dispatch_branches_on(self) -> None:
        """The refusal below is only worth having while our copy of the names is TRL's own.

        Read out of the dispatch itself (`grpo_trainer.py:2653-2692`) rather than out of the help
        text, because the help text is where TRL's own copy has already drifted: `grpo_config.py:338`
        calls `token_truncate` the default while the field's default is `sequence_mask`.
        """
        dispatched = {
            name.strip().strip("\"'")
            for group in re.findall(
                r"vllm_importance_sampling_mode in \[([^\]]+)\]",
                inspect.getsource(gt.GRPOTrainer),
            )
            for name in group.split(",")
        }
        assert dispatched == set(VLLM_IMPORTANCE_SAMPLING_MODES)

    def test_the_mode_lands_in_the_run_record(self) -> None:
        assert asdict(make_config())["vllm_importance_sampling_mode"] == (
            VLLM_IMPORTANCE_SAMPLING_MODE
        )
        assert (
            asdict(make_config(vllm_importance_sampling_mode="token_mask"))[
                "vllm_importance_sampling_mode"
            ]
            == "token_mask"
        )

    @pytest.mark.parametrize("mode", VLLM_IMPORTANCE_SAMPLING_MODES)
    def test_every_mode_trl_dispatches_on_is_accepted(self, mode: str) -> None:
        assert make_config(vllm_importance_sampling_mode=mode).vllm_importance_sampling_mode == mode

    def test_a_misspelled_mode_is_refused_at_startup(self) -> None:
        """TRL raises on an unknown name inside the first generation batch's ratio arithmetic, which
        on a rented card is the bootstrap, the model pull and a generation pass too late."""
        with pytest.raises(ValueError, match="token_truncate"):
            make_config(vllm_importance_sampling_mode="token-truncate")

    def test_a_non_default_mode_without_the_correction_is_refused(self) -> None:
        """There is no ratio to compute with the correction off, so the mode would select nothing
        while `run_config.json` recorded a treatment the run never ran."""
        with pytest.raises(ValueError, match="--vllm-importance-sampling-mode"):
            make_config(
                vllm_importance_sampling_mode="token_truncate",
                vllm_importance_sampling_correction=False,
            )

    def test_the_correction_off_carries_trls_default_mode_without_complaint(self) -> None:
        """A correction-off run records the field too, at the value nothing read: refusing that
        would refuse every correction-off arm."""
        config = make_config(vllm_importance_sampling_correction=False)
        assert config.vllm_importance_sampling_mode == VLLM_IMPORTANCE_SAMPLING_MODE

    def test_the_flag_is_spelled_the_way_the_kit_will_spell_it(self) -> None:
        config = gt._parse_args(  # pyright: ignore[reportPrivateUsage]
            [
                "--arm",
                "twin-pd-group",
                "--generate-fresh",
                "--vllm-importance-sampling-mode",
                "token_truncate",
            ]
        )
        assert config.vllm_importance_sampling_mode == "token_truncate"

    def test_the_command_line_default_is_the_dataclass_default(self) -> None:
        config = gt._parse_args(["--arm", "twin-pd-group", "--generate-fresh"])  # pyright: ignore[reportPrivateUsage]
        assert config.vllm_importance_sampling_mode == VLLM_IMPORTANCE_SAMPLING_MODE

    def test_the_help_carries_the_owners_precommitted_rule(self) -> None:
        """The rule is a decision about arm 1, so it belongs where the operator reads the flag.

        Whitespace-normalised because argparse rewraps help text to the terminal width, so any phrase
        long enough to be worth asserting on can arrive split across a line break.
        """
        parser = gt.argparse.ArgumentParser()
        gt._add_sampler_mismatch_arguments(parser)  # pyright: ignore[reportPrivateUsage]
        help_text = " ".join(parser.format_help().split())
        assert "token_truncate" in help_text
        assert "arm 1 and its control run if the 9B probe confirms a large mismatch" in help_text

    @pytest.mark.parametrize("mode", VESPO_IMPORTANCE_SAMPLING_MODES)
    def test_vespo_becomes_runnable_under_the_token_level_modes_trl_requires(
        self, mode: str
    ) -> None:
        """TRL refuses vespo with the correction ON unless the mode is token-level
        (`grpo_trainer.py:924-928`). That pair was unreachable while the mode was fixed, and the
        refusal below has to narrow to what TRL actually refuses rather than to the whole loss type.
        """
        config = make_config(
            loss_type="vespo",
            acknowledge_liger_estimator_mismatch=True,
            vllm_importance_sampling_mode=mode,
        )
        assert config.loss_type == "vespo"
        assert config.vllm_importance_sampling_correction is True

    @pytest.mark.parametrize(
        "mode",
        [
            mode
            for mode in VLLM_IMPORTANCE_SAMPLING_MODES
            if mode not in VESPO_IMPORTANCE_SAMPLING_MODES
        ],
    )
    def test_vespo_under_a_sequence_level_mode_is_still_refused(self, mode: str) -> None:
        with pytest.raises(ValueError, match="vespo"):
            make_config(
                loss_type="vespo",
                acknowledge_liger_estimator_mismatch=True,
                vllm_importance_sampling_mode=mode,
            )

    def test_the_vespo_requirement_is_the_one_trl_states(self) -> None:
        """Our copy of the pair, read out of TRL's own refusal rather than trusted."""
        required = re.search(
            r"vllm_importance_sampling_mode not in \[([^\]]+)\]", inspect.getsource(gt.GRPOTrainer)
        )
        assert required is not None
        assert {name.strip().strip("\"'") for name in required.group(1).split(",")} == set(
            VESPO_IMPORTANCE_SAMPLING_MODES
        )


class TestTheTrainerIsBuiltWithTheKnobsTheConfigCarries:
    def test_the_class_defaults_keep_a_caller_that_passes_neither_on_todays_behaviour(self) -> None:
        """`reward_hacking.train` builds this class without either keyword; it must not change."""
        assert gt.InstrumentedGRPOTrainer.old_logps_chunk_tokens == gt.OLD_LOGPS_CHUNK_TOKENS
        assert gt.InstrumentedGRPOTrainer.importance_sampling_log_only is False

    def test_padding_trim_still_extends_the_instrumented_trainer(self) -> None:
        assert issubclass(gt.PaddingTrimmedGRPOTrainer, gt.InstrumentedGRPOTrainer)
        assert issubclass(gt.InstrumentedGRPOTrainer, gt.GRPOTrainer)


SEAM_HIDDEN_SIZE = 32
SEAM_VOCAB_SIZE = 64


def tiny_peft_causal_lm(*, adapter: bool = True) -> torch.nn.Module:
    """A one-layer causal LM with a real `lm_head`, optionally behind a LoRA adapter.

    Built from a `Qwen2Config` directly rather than loaded, so this needs no hub read and no cache.
    The adapter is what makes the seam interesting: the override reaches the head as
    `unwrapped.lm_head`, which on a `PeftModel` resolves through `__getattr__` two levels down, and
    TRL's `_get_last_hidden_state` takes its own `is_peft_model` branch to find the backbone.
    """
    config = Qwen2Config(
        vocab_size=SEAM_VOCAB_SIZE,
        hidden_size=SEAM_HIDDEN_SIZE,
        intermediate_size=2 * SEAM_HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    if adapter:
        model = get_peft_model(
            model,
            LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]),
        )
    return model.eval()


def seam_trainer(
    *, chunk_tokens: int = 3, temperature: float = 1.0
) -> gt.PaddingTrimmedGRPOTrainer:
    """A bare trainer carrying exactly what the old-logps pass reads, its own and TRL's.

    The `__new__` pattern `test_games_padding_trim.py` established: no `GRPOTrainer.__init__`, no
    model load, no accelerator. `model_kwarg_keys` is empty on purpose, which is the branch where TRL
    does NOT pass `logits_to_keep` to the model and slices the positions itself -- the arithmetic the
    override has to reproduce. `state`, `args.report_to` and `accelerator.is_main_process` are there
    for `profiling_decorator`, which wraps both TRL methods this exercises.
    """
    trainer = gt.PaddingTrimmedGRPOTrainer.__new__(gt.PaddingTrimmedGRPOTrainer)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model, is_main_process=True)  # pyright: ignore[reportAttributeAccessIssue]
    trainer.state = SimpleNamespace(global_step=0)  # pyright: ignore[reportAttributeAccessIssue]
    trainer.args = SimpleNamespace(report_to=[])  # pyright: ignore[reportAttributeAccessIssue]
    trainer.model_kwarg_keys = set()  # pyright: ignore[reportAttributeAccessIssue]
    trainer._is_vlm = False  # pyright: ignore[reportAttributeAccessIssue]
    trainer._entropy_bonus_enabled = False  # pyright: ignore[reportAttributeAccessIssue]
    trainer.temperature = temperature
    trainer.old_logps_chunk_tokens = chunk_tokens
    return trainer


def seam_inputs(*, rows: int = 2, positions: int = 9) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Prompt-plus-completion ids, an all-live attention mask, and how many logits TRL would keep."""
    generator = torch.Generator().manual_seed(3)
    input_ids = torch.randint(1, SEAM_VOCAB_SIZE, (rows, positions), generator=generator)
    return input_ids, torch.ones_like(input_ids), positions - 4


class TestTheTrainerSeamIsTheChunkedPass:
    """The override that actually replaces TRL's unchunked old-log-probability pass.

    Everything above drives `chunked_per_token_logps` directly, which leaves the wiring untested: the
    seam has to find the head through a `PeftModel`, hand `_get_last_hidden_state` TRL's own positional
    arguments, slice `input_ids` to the kept logits, and split its own row groups. If any of that
    regresses -- TRL reorders `_get_last_hidden_state`'s parameters, PEFT stops forwarding `.lm_head`,
    a slice goes off by one -- the first old-logps pass after a paid generation batch either raises or
    silently falls back to the 30.31 GiB pass and dies mid-run on a rented box.

    Sabotage run once and watched red (2026-09-04): shifting the sampled-id slice in the override one
    position earlier than the hidden-state slice failed three of the five tests here.
    """

    FP32_TOLERANCE_NATS = 1e-4

    def test_the_override_computes_what_trls_own_pass_computes(self) -> None:
        """Head applied to a slice of hidden states against head applied to the whole row, on one
        model, through both code paths. Agreement is floating point, not bit, for the reason
        `chunked_per_token_logps` documents: the matmul reassociates under a change of row count."""
        model = tiny_peft_causal_lm()
        input_ids, attention_mask, logits_to_keep = seam_inputs()
        trainer = seam_trainer()
        with torch.no_grad():
            chunked, entropies, aux_loss = trainer._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                model, input_ids, attention_mask, logits_to_keep
            )
            whole, _, _ = gt.GRPOTrainer._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                trainer, model, input_ids, attention_mask, logits_to_keep
            )
        assert entropies is None
        assert aux_loss is None
        assert chunked.shape == (input_ids.size(0), logits_to_keep)
        assert torch.allclose(chunked, whole, rtol=0.0, atol=self.FP32_TOLERANCE_NATS)

    def test_the_head_is_only_ever_asked_for_a_chunk_of_positions(self) -> None:
        """The claim the memory refusal rests on, made at the seam rather than on the free function."""
        model = tiny_peft_causal_lm()
        input_ids, attention_mask, logits_to_keep = seam_inputs()
        widths: list[int] = []
        inner = cast("Any", model).base_model.model
        head = inner.lm_head

        class RecordingHead(torch.nn.Module):
            def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                widths.append(hidden.size(1))
                return cast("torch.Tensor", head(hidden))

        inner.lm_head = RecordingHead()
        trainer = seam_trainer(chunk_tokens=2)
        with torch.no_grad():
            trainer._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                model, input_ids, attention_mask, logits_to_keep
            )
        assert widths == [2, 2, 1]
        assert max(widths) <= 2

    def test_the_temperature_reaches_the_logits_through_the_seam(self) -> None:
        """A temperature elsewhere than 1.0 must divide the logits, not the log-probabilities."""
        model = tiny_peft_causal_lm()
        input_ids, attention_mask, logits_to_keep = seam_inputs()
        with torch.no_grad():
            tempered, _, _ = seam_trainer(temperature=0.7)._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                model, input_ids, attention_mask, logits_to_keep
            )
            plain, _, _ = seam_trainer()._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                model, input_ids, attention_mask, logits_to_keep
            )
            reference, _, _ = gt.GRPOTrainer._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                seam_trainer(temperature=0.7), model, input_ids, attention_mask, logits_to_keep
            )
        assert not torch.allclose(tempered, plain, rtol=0.0, atol=1e-3)
        assert torch.allclose(tempered, reference, rtol=0.0, atol=self.FP32_TOLERANCE_NATS)

    def test_the_row_groups_are_concatenated_in_the_order_trl_expects(self) -> None:
        """TRL hands `batch_size` rows at a time; a per-group result has to land back in row order."""
        model = tiny_peft_causal_lm()
        input_ids, attention_mask, logits_to_keep = seam_inputs(rows=4)
        trainer = seam_trainer()
        with torch.no_grad():
            grouped, _, _ = trainer._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                model, input_ids, attention_mask, logits_to_keep, 2
            )
            whole, _, _ = trainer._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                model, input_ids, attention_mask, logits_to_keep
            )
        assert grouped.shape == whole.shape
        assert torch.allclose(grouped, whole, rtol=0.0, atol=self.FP32_TOLERANCE_NATS)

    def test_an_unadapted_model_goes_down_the_same_seam(self) -> None:
        """Both threads build with an adapter, but the override must not depend on one:
        `unwrapped.lm_head` has to resolve either way."""
        model = tiny_peft_causal_lm(adapter=False)
        input_ids, attention_mask, logits_to_keep = seam_inputs()
        trainer = seam_trainer()
        with torch.no_grad():
            chunked, _, _ = trainer._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                model, input_ids, attention_mask, logits_to_keep
            )
            whole, _, _ = gt.GRPOTrainer._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                trainer, model, input_ids, attention_mask, logits_to_keep
            )
        assert torch.allclose(chunked, whole, rtol=0.0, atol=self.FP32_TOLERANCE_NATS)


class TestTheSeamHandsBackWhatASliceCannotServe:
    """Three callers need the full-vocabulary logits, and each has to reach TRL's own pass untouched.

    `super()` is stubbed rather than run, the pattern `bare_trainer` uses above: the point under test
    is which caller routes where, and TRL's own pass on a dense one-layer stand-in cannot serve the
    mixture-of-experts branch at all (it asks the model for router logits and reads `outputs.aux_loss`).
    """

    SENTINEL: ClassVar[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.full((1,), 7.0),
    )

    @pytest.fixture(autouse=True)
    def trl_pass_stubbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stand in for TRL's unchunked pass, so "handed back" is observable without running it."""
        monkeypatch.setattr(
            gt.GRPOTrainer,
            "_get_per_token_logps_and_entropies",
            lambda *_args, **_kwargs: self.SENTINEL,
        )

    def call(self, caplog: pytest.LogCaptureFixture, **kwargs: Any) -> object:
        model = tiny_peft_causal_lm()
        input_ids, attention_mask, logits_to_keep = seam_inputs()
        with caplog.at_level(logging.INFO, logger="games.train"), torch.no_grad():
            return seam_trainer()._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                model, input_ids, attention_mask, logits_to_keep, None, **kwargs
            )

    def test_entropies_hand_the_pass_back(self, caplog: pytest.LogCaptureFixture) -> None:
        """The non-Liger loss path asks for them, and it is building the full logits anyway."""
        assert self.call(caplog, compute_entropy=True) is self.SENTINEL
        assert "left unchunked" in caplog.text
        assert "compute_entropy=True" in caplog.text

    def test_the_moe_auxiliary_loss_hands_the_pass_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """It needs router logits from the model's own forward, which a hidden state cannot give."""
        assert self.call(caplog, compute_aux_loss=True) is self.SENTINEL
        assert "compute_aux_loss=True" in caplog.text

    def test_a_multimodal_input_hands_the_pass_back(self, caplog: pytest.LogCaptureFixture) -> None:
        """`_get_last_hidden_state` takes pixels but not the image counts the wider signature has."""
        assert self.call(caplog, pixel_values=torch.zeros(1, 3)) is self.SENTINEL
        assert "multimodal inputs ['pixel_values']" in caplog.text

    def test_an_absent_multimodal_keyword_does_not_count_as_supplied(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """TRL's own call site passes every multimodal keyword as None on a text-only run."""
        result = self.call(caplog, pixel_values=None, image_grid_thw=None, token_type_ids=None)
        assert result is not self.SENTINEL
        assert "left unchunked" not in caplog.text

    def test_the_reason_is_said_once_per_run(self, caplog: pytest.LogCaptureFixture) -> None:
        model = tiny_peft_causal_lm()
        input_ids, attention_mask, logits_to_keep = seam_inputs()
        trainer = seam_trainer()
        with caplog.at_level(logging.INFO, logger="games.train"), torch.no_grad():
            for _ in range(3):
                trainer._get_per_token_logps_and_entropies(  # pyright: ignore[reportPrivateUsage]
                    model, input_ids, attention_mask, logits_to_keep, None, True
                )
        assert caplog.text.count("left unchunked") == 1


def liger_style_per_token_logps(
    head: torch.nn.Linear, hidden: torch.Tensor, token_ids: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    """The per-token log-probabilities Liger's fused GRPO loss computes for the same head.

    Written out because it is the claim under test rather than a re-implementation for convenience.
    TRL hands Liger the raw `lm_head.weight` instead of calling `lm_head.forward`
    (`grpo_trainer.py:2951-2956`), Liger casts each vocab chunk of that weight back to the hidden
    dtype before the matmul and only then upcasts (`fused_linear_ppo.py:47`), the bias is added in
    fp32 (`:48-49`) and the inverse temperature multiplied in (`:50`), and the softmax is accumulated
    in fp32. So the loss's own logits come off bfloat16 weights whether or not TRL's fp32-head knob
    is set -- which is the half of `cast_lm_head_to_fp32`'s documentation that used to be wrong. The
    head this is called with always carries a bias, so there is no branch for one that does not.
    """
    logits = (hidden @ head.weight.to(hidden.dtype).t()).float() + head.bias.to(torch.float32)
    return selective_log_softmax(logits / temperature, token_ids)


class TestTheFp32HeadMovesTheOldLogpsPassTowardTheLigerLoss:
    """Why `cast_lm_head_to_fp32` is NOT refused alongside a gradient-moving correction.

    The hazard to rule out: with the correction ON and log-only off, TRL's surrogate ratio is
    `exp(per_token_logps - old_per_token_logps)`, the first from the Liger loss and the second from
    this trainer's chunked pass. If the fp32 head moved those two apart, a nominally on-policy run
    would be training on head-rounding noise and clipping on it.

    It moves them together instead, and for a reason the numbers below show: Liger's logits are a
    bfloat16 matmul upcast to fp32, so an fp32 old-logps pass differs from them only by that rounding,
    while a bfloat16 pass ALSO takes `selective_log_softmax`'s bfloat16 branch and differs by the
    log-softmax as well. Measured here on a CPU stand-in for the two arithmetics rather than on the
    production model, so the direction is the claim and the magnitudes are not.

    Sabotage run once and watched red (2026-09-04): making the stand-in call
    `chunked_per_token_logps` with the fp32 head -- so the comparison compared the pass with itself --
    failed the teeth test below.
    """

    HIDDEN, VOCAB, POSITIONS, ROWS = 64, 512, 12, 3

    def head_and_hidden(self) -> tuple[torch.nn.Linear, torch.Tensor, torch.Tensor]:
        """A bfloat16 head and bfloat16 hidden states, the dtypes a production run's loss sees."""
        generator = torch.Generator().manual_seed(11)
        head = torch.nn.Linear(self.HIDDEN, self.VOCAB, bias=True, dtype=torch.bfloat16)
        with torch.no_grad():
            head.weight.copy_(
                torch.randn(self.VOCAB, self.HIDDEN, generator=generator).to(torch.bfloat16)
            )
            head.bias.copy_(torch.randn(self.VOCAB, generator=generator).to(torch.bfloat16))
        hidden = torch.randn(self.ROWS, self.POSITIONS, self.HIDDEN, generator=generator).to(
            torch.bfloat16
        )
        token_ids = torch.randint(0, self.VOCAB, (self.ROWS, self.POSITIONS), generator=generator)
        return head, hidden, token_ids

    def fp32_cast_head(self, head: torch.nn.Linear) -> Callable[[torch.Tensor], torch.Tensor]:
        """TRL's replacement forward under `cast_lm_head_to_fp32` (`grpo_trainer.py:1007-1012`)."""

        def forward(hidden: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(
                hidden.to(torch.float32),
                head.weight.to(torch.float32),
                head.bias.to(torch.float32),
            )

        return forward

    def test_the_fp32_head_agrees_with_the_liger_loss_more_closely_than_bfloat16(self) -> None:
        head, hidden, token_ids = self.head_and_hidden()
        liger = liger_style_per_token_logps(head, hidden, token_ids, temperature=1.0)
        bfloat16_pass = chunked_per_token_logps(
            hidden, head, token_ids, temperature=1.0, chunk_tokens=3
        ).float()
        fp32_pass = chunked_per_token_logps(
            hidden, self.fp32_cast_head(head), token_ids, temperature=1.0, chunk_tokens=3
        )
        bfloat16_gap = float((bfloat16_pass - liger).abs().mean().detach())
        fp32_gap = float((fp32_pass - liger).abs().mean().detach())
        logger.info("old-logps against the liger loss: %s", f"{bfloat16_gap=:.5f} {fp32_gap=:.5f}")
        assert fp32_gap < bfloat16_gap

    def test_the_liger_arithmetic_is_not_the_pass_it_is_compared_with(self) -> None:
        """Teeth: were the stand-in the same code as the pass, the comparison above would be empty."""
        head, hidden, token_ids = self.head_and_hidden()
        liger = liger_style_per_token_logps(head, hidden, token_ids, temperature=1.0)
        fp32_pass = chunked_per_token_logps(
            hidden, self.fp32_cast_head(head), token_ids, temperature=1.0, chunk_tokens=3
        )
        assert not torch.equal(fp32_pass, liger)


class TestTheTemperatureDivisionDoesNotCopyTheLogits:
    """The peak the memory refusal prices assumes one divided fp32 copy per row, not two.

    TRL writes `logits = logits / self.temperature`, holding the undivided slice alive beside its
    quotient at exactly the moment the tensor is widest. `_tempered_logits` divides in place instead,
    and skips the division entirely at temperature 1.0, which is bit-identical because IEEE-754
    division by 1.0 is exact.

    Sabotage run once and watched red (2026-09-04): restoring TRL's copying `head_output / temperature`
    failed two of the three tests here.
    """

    def test_temperature_one_returns_the_heads_own_tensor_untouched(self) -> None:
        head, hidden, _ = head_and_inputs()
        logits = head(hidden)
        assert gt._tempered_logits(logits, 1.0) is logits  # pyright: ignore[reportPrivateUsage]

    def test_another_temperature_divides_the_same_storage(self) -> None:
        head, hidden, _ = head_and_inputs()
        logits = head(hidden)
        expected = logits.detach().clone() / 0.7
        divided = gt._tempered_logits(logits, 0.7)  # pyright: ignore[reportPrivateUsage]
        assert divided is logits
        assert torch.equal(divided.detach(), expected)

    def test_the_in_place_division_leaves_the_gradient_bit_identical(self) -> None:
        """A matmul's backward does not save its output, so the in-place divide is autograd-neutral.

        Run at a chunk width of the whole row, so the ONLY difference from TRL's copying division is
        the in-place one: a narrower chunk would also reassociate the head's matmul, which moved this
        gradient by 1.9e-6 when the comparison was written that way and would have made a bit
        assertion impossible for a reason that has nothing to do with the division.
        """
        head, hidden, token_ids = head_and_inputs()
        chunked_per_token_logps(
            hidden, head, token_ids, temperature=0.7, chunk_tokens=hidden.size(1)
        ).sum().backward()
        in_place_weight = cast("torch.Tensor", head.weight.grad).clone()
        in_place_bias = cast("torch.Tensor", head.bias.grad).clone()
        head.zero_grad()
        selective_log_softmax(head(hidden) / 0.7, token_ids).sum().backward()
        assert torch.equal(in_place_weight, cast("torch.Tensor", head.weight.grad))
        assert torch.equal(in_place_bias, cast("torch.Tensor", head.bias.grad))
