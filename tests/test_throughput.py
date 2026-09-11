"""Tests for the GRPO throughput harness, CPU only and no model downloads.

Half of these are sabotage tests. `grpo/throughput.py` carries three guards whose whole
purpose is to stop a number being published under a label it does not deserve — a LoRA config
that froze three quarters of the model, a prompt that came out shorter than requested, a
completion that stopped early so the step time belongs to a cheaper profile than the one named.
A guard nobody has watched fail is not a guard, so each one is exercised here by feeding it the
exact condition it exists to catch.
"""

from __future__ import annotations

import collections
import math
import types

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from grpo import throughput as tp
from grpo import throughput_sweep

# Qwen/Qwen3.5-4B, read from its config.json on 2026-08-15. Duplicated here rather than
# downloaded so the arithmetic these tests pin is checkable offline.
QWEN35_4B_LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]


def qwen35_4b_text_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        layer_types=QWEN35_4B_LAYER_TYPES * 8,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        vocab_size=248320,
        hidden_size=2560,
    )


class StubTokenizer:
    """Counts a fixed template overhead plus one token per filler occurrence.

    `merge_loss` simulates the real tokenizer merging a byte pair at the boundary between the
    template and the body, which is the case `build_prompt`'s correction pass exists for.
    `constant_tokens` simulates a tokenizer whose count does not respond to the body at all,
    which must be detected rather than silently accepted.
    """

    def __init__(
        self, overhead: int = 7, merge_loss: int = 0, constant_tokens: int | None = None
    ) -> None:
        self.overhead = overhead
        self.merge_loss = merge_loss
        self.constant_tokens = constant_tokens
        self.render_calls = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        self.render_calls += 1
        return "TEMPLATE" + messages[-1]["content"]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        if self.constant_tokens is not None:
            return {"input_ids": [0] * self.constant_tokens}
        fillers = text.count(" apple")
        n = self.overhead + fillers - (self.merge_loss if fillers else 0)
        return {"input_ids": [0] * max(n, 0)}


def fake_trainer(log_history: list[dict[str, object]]) -> types.SimpleNamespace:
    return types.SimpleNamespace(state=types.SimpleNamespace(log_history=log_history))


def healthy_log_record(completion_length: float = 2048.0) -> dict[str, object]:
    return {
        "reward": 0.21,
        "reward_std": 0.04,
        "frac_reward_zero_std": 0.0,
        "completions/mean_length": completion_length,
        "completions/min_length": completion_length,
        "completions/max_length": completion_length,
        "step_time": 511.3,
    }


class TestBatchingArithmetic:
    """The mapping from prompts-per-step and group size onto what TRL actually consumes.

    TRL derives `generation_batch_size` as `per_device_train_batch_size *
    gradient_accumulation_steps`, and splits it into that many divided by `num_generations`
    unique prompts. Getting this wrong is silent: the run trains happily on one prompt per
    step and the label on the measurement is simply false.
    """

    def test_episodes_per_step_is_prompts_times_group(self):
        config = tp.ThroughputConfig(model_id="stub", prompts_per_step=8, group_size=8)
        assert config.episodes_per_step == 64

    def test_micro_batch_times_accumulation_equals_episodes(self):
        config = tp.ThroughputConfig(model_id="stub", prompts_per_step=3, group_size=8)
        assert config.resolved_micro_batch_size == 8
        assert config.gradient_accumulation_steps == 3
        assert config.resolved_micro_batch_size * config.gradient_accumulation_steps == 24

    def test_explicit_micro_batch_is_honoured(self):
        config = tp.ThroughputConfig(
            model_id="stub", prompts_per_step=8, group_size=8, micro_batch_size=4
        )
        assert config.gradient_accumulation_steps == 16
        assert config.resolved_micro_batch_size * config.gradient_accumulation_steps == 64

    def test_validate_rejects_micro_batch_that_does_not_divide(self):
        config = tp.ThroughputConfig(
            model_id="stub", prompts_per_step=3, group_size=8, micro_batch_size=5
        )
        with pytest.raises(ValueError, match="must divide"):
            config.validate()

    def test_validate_rejects_group_size_below_two(self):
        with pytest.raises(ValueError, match="at least 2 generations"):
            tp.ThroughputConfig(model_id="stub", group_size=1).validate()

    def test_validate_refuses_a_liger_unfaithful_loss_constant(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A throughput number under the wrong executed estimator mislabels every comparison, so
        an edit of GRPO_LOSS_TYPE to a value Liger degrades must refuse at validate()."""
        monkeypatch.setattr(tp, "GRPO_LOSS_TYPE", "dapo")
        with pytest.raises(ValueError, match="num_items_in_batch"):
            tp.ThroughputConfig(model_id="stub").validate()
        tp.ThroughputConfig(model_id="stub", acknowledge_liger_estimator_mismatch=True).validate()
        tp.ThroughputConfig(model_id="stub", use_liger_kernel=False).validate()

    def test_total_steps_includes_warmup(self):
        config = tp.ThroughputConfig(model_id="stub", warmup_steps=1, measured_steps=3)
        assert config.total_steps == 4


class TestMemoryModel:
    """Pins the config-derived arithmetic that `docs/scratch/measured-throughput.md` publishes."""

    def test_kv_bytes_per_token_counts_only_full_attention_layers(self):
        prediction = tp.memory_model(
            qwen35_4b_text_config(), episodes=64, prompt_tokens=2048, completion_tokens=2048
        )
        assert prediction["n_full_attention_layers"] == 8
        assert prediction["n_linear_attention_layers"] == 24
        # 2 (K and V) x 8 layers x 4 heads x 256 dim x 2 bytes.
        assert prediction["kv_bytes_per_token"] == 32768

    def test_recurrent_state_is_float32_and_constant_in_context(self):
        short = tp.memory_model(
            qwen35_4b_text_config(), episodes=16, prompt_tokens=512, completion_tokens=512
        )
        long = tp.memory_model(
            qwen35_4b_text_config(), episodes=16, prompt_tokens=16384, completion_tokens=512
        )
        # 24 layers x 32 value heads x 128 x 128 x 4 bytes = 50,331,648 B = 48.0 MiB. The
        # budget note's "~25 MB per sequence" assumed the model dtype; transformers keeps this
        # state in float32, so the real figure is twice that.
        assert short["recurrent_state_mib_per_sequence"] == pytest.approx(48.0, abs=0.01)
        assert short["predicted_recurrent_gib"] == long["predicted_recurrent_gib"], (
            "the recurrent state must not scale with context length"
        )
        assert (
            long["predicted_kv_gib"] > short["predicted_kv_gib"]  # pyright: ignore[reportOperatorIssue]
        )

    def test_kv_scales_linearly_with_episodes_and_tokens(self):
        base = tp.memory_model(
            qwen35_4b_text_config(), episodes=8, prompt_tokens=1024, completion_tokens=1024
        )
        doubled = tp.memory_model(
            qwen35_4b_text_config(), episodes=16, prompt_tokens=1024, completion_tokens=1024
        )
        assert doubled["predicted_kv_gib"] == pytest.approx(
            2 * base["predicted_kv_gib"]  # pyright: ignore[reportOperatorIssue]
        )


class TestLoraTargetSelection:
    def test_finds_both_attention_families_and_the_mlp(self):
        names = [
            "model.layers.0.linear_attn.in_proj_qkv",
            "model.layers.0.linear_attn.in_proj_z",
            "model.layers.0.linear_attn.in_proj_b",
            "model.layers.0.linear_attn.in_proj_a",
            "model.layers.0.linear_attn.out_proj",
            "model.layers.3.self_attn.q_proj",
            "model.layers.3.self_attn.k_proj",
            "model.layers.3.self_attn.v_proj",
            "model.layers.3.self_attn.o_proj",
            "model.layers.3.mlp.gate_proj",
            "lm_head",
        ]
        selected = tp.select_lora_targets(names, QWEN35_4B_LAYER_TYPES)
        assert (
            "in_proj_qkv" in selected["target_modules"]  # pyright: ignore[reportOperatorIssue]
        )
        assert "q_proj" in selected["target_modules"]  # pyright: ignore[reportOperatorIssue]
        assert (
            "lm_head" not in selected["target_modules"]  # pyright: ignore[reportOperatorIssue]
        ), "TRL refuses to start with a head adapter under use_liger_kernel"
        assert selected["expected_linear_attention_layers"] == 3
        assert selected["expected_full_attention_layers"] == 1

    def test_ignores_vision_tower_projections(self):
        names = [
            "model.language_model.layers.3.self_attn.q_proj",
            "model.visual.blocks.0.attn.qkv",
            "model.visual.blocks.0.attn.proj",
            "model.visual.blocks.0.mlp.linear_fc1",
            "model.layers.0.linear_attn.in_proj_qkv",
        ]
        selected = tp.select_lora_targets(names, QWEN35_4B_LAYER_TYPES)
        assert selected["module_counts"] == {"in_proj_qkv": 1, "q_proj": 1}

    def test_raises_when_linear_attention_projections_are_absent(self):
        """The sabotage case: the familiar target list on a hybrid-attention model.

        This is not hypothetical — that list is what a training script writes by hand, and on
        Qwen3.5 it would adapt 8 layers of 32 while reporting a healthy run. `grpo/rlvr_math.py`
        used to ship exactly it and now calls `discover_lora_targets`, so this guard is what
        stands between a hand-written list and three quarters of the stack frozen.
        """
        names = [
            "model.layers.3.self_attn.q_proj",
            "model.layers.3.self_attn.k_proj",
            "model.layers.3.mlp.gate_proj",
        ]
        with pytest.raises(RuntimeError, match="freeze them silently"):
            tp.select_lora_targets(names, QWEN35_4B_LAYER_TYPES)

    def test_accepts_a_model_with_no_linear_attention_layers(self):
        selected = tp.select_lora_targets(
            ["model.layers.0.self_attn.q_proj"], ["full_attention", "full_attention"]
        )
        assert selected["module_counts"] == {"q_proj": 1}

    def test_a_dense_qwen3_yields_exactly_the_familiar_seven(self):
        """What `grpo/rlvr_math.py`'s toy default gets, now that it discovers rather than declares.

        Built from a hand-made small config on the meta device, so this pins the
        behaviour-preservation claim without a download: on a dense Qwen3, discovery returns the
        same seven names the training script used to hardcode, so no measured baseline moves.
        """
        config = Qwen3Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        )
        layer_types = list(config.get_text_config().layer_types)
        with torch.device("meta"):
            model = Qwen3ForCausalLM(config)
        names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

        selected = tp.select_lora_targets(names, layer_types)
        assert selected["target_modules"] == [
            "down_proj",
            "gate_proj",
            "k_proj",
            "o_proj",
            "q_proj",
            "up_proj",
            "v_proj",
        ]


class TestPromptConstruction:
    def test_hits_the_requested_token_count_exactly(self):
        tokenizer = StubTokenizer(overhead=7)
        text, measured = tp.build_prompt(
            tokenizer,  # pyright: ignore[reportArgumentType]
            target_tokens=100,
            filler=" apple",
        )
        assert measured == 100
        assert len(tokenizer(text)["input_ids"]) == 100

    def test_correction_pass_recovers_a_boundary_merge(self):
        tokenizer = StubTokenizer(overhead=7, merge_loss=2)
        _text, measured = tp.build_prompt(
            tokenizer,  # pyright: ignore[reportArgumentType]
            target_tokens=100,
            filler=" apple",
        )
        assert measured == 100
        assert tokenizer.render_calls == 3, "expected overhead probe, first attempt, correction"

    def test_raises_rather_than_returning_a_short_prompt(self):
        """A prompt that silently came out short would mislabel the whole measurement."""
        tokenizer = StubTokenizer(constant_tokens=40)
        with pytest.raises(RuntimeError, match="could not hit an exact prompt length"):
            tp.build_prompt(
                tokenizer,  # pyright: ignore[reportArgumentType]
                target_tokens=100,
                filler=" apple",
            )

    def test_raises_when_the_template_alone_exceeds_the_budget(self):
        tokenizer = StubTokenizer(overhead=200)
        with pytest.raises(ValueError, match="chat template alone"):
            tp.build_prompt(
                tokenizer,  # pyright: ignore[reportArgumentType]
                target_tokens=100,
                filler=" apple",
            )

    def test_verify_single_token_drops_multi_token_fillers(self):
        class PerWordTokenizer:
            def __call__(self, text: str, add_special_tokens: bool = False):
                del add_special_tokens
                return {"input_ids": [0] * (2 if text == " basket" else 1)}

        kept = tp.verify_single_token(
            PerWordTokenizer(),  # pyright: ignore[reportArgumentType]
            (" apple", " basket"),
        )
        assert kept == [" apple"]

    def test_verify_single_token_raises_when_none_qualify(self):
        class WideTokenizer:
            def __call__(self, text: str, add_special_tokens: bool = False):
                del add_special_tokens
                return {"input_ids": [0, 0, 0]}

        with pytest.raises(RuntimeError, match="no single-token filler"):
            tp.verify_single_token(
                WideTokenizer(),  # pyright: ignore[reportArgumentType]
                (" apple",),
            )


class TestMetricVerification:
    """Guards against the failure this repo has already been bitten by once.

    A callback wrote metrics into the dict `Trainer.log` had already copied, so logging ran
    clean and recorded nothing for the entire history of the repo. Reading the metrics back out
    of trainer state and refusing to report without them is the fix; these tests are what make
    it a check rather than a hope.
    """

    def test_accepts_a_complete_record(self):
        config = tp.ThroughputConfig(model_id="stub", completion_tokens=2048)
        metrics = tp.verify_logged_metrics(
            fake_trainer([healthy_log_record()]),  # pyright: ignore[reportArgumentType]
            config,
        )
        assert metrics["completions_mean_length"] == 2048.0
        assert metrics["frac_reward_zero_std"] == 0.0
        assert metrics["trl_step_time_mean"] == pytest.approx(511.3)

    def test_raises_when_completions_stopped_early(self):
        """min_new_tokens failing to suppress EOS must not pass as the requested profile."""
        config = tp.ThroughputConfig(model_id="stub", completion_tokens=2048)
        short = healthy_log_record(completion_length=311.0)
        with pytest.raises(RuntimeError, match="did not run to the requested length"):
            tp.verify_logged_metrics(
                fake_trainer([short]),  # pyright: ignore[reportArgumentType]
                config,
            )

    def test_raises_when_an_expected_metric_never_logged(self):
        config = tp.ThroughputConfig(model_id="stub", completion_tokens=2048)
        record = healthy_log_record()
        del record["frac_reward_zero_std"]
        with pytest.raises(RuntimeError, match="never reached trainer_state"):
            tp.verify_logged_metrics(
                fake_trainer([record]),  # pyright: ignore[reportArgumentType]
                config,
            )

    def test_raises_when_no_step_logged_a_reward(self):
        config = tp.ThroughputConfig(model_id="stub")
        with pytest.raises(RuntimeError, match="nothing was measured"):
            tp.verify_logged_metrics(
                fake_trainer([{"loss": 0.0}]),  # pyright: ignore[reportArgumentType]
                config,
            )

    def test_reads_the_last_reward_record_not_the_summary(self):
        config = tp.ThroughputConfig(model_id="stub", completion_tokens=2048)
        history = [
            healthy_log_record() | {"reward": 0.1},
            healthy_log_record() | {"reward": 0.9},
            {"train_runtime": 37.2},
        ]
        metrics = tp.verify_logged_metrics(
            fake_trainer(history),  # pyright: ignore[reportArgumentType]
            config,
        )
        assert metrics["reward"] == pytest.approx(0.9)


class PhaseClock:
    """A `perf_counter` the fake trainer advances by hand, so every phase comes out an exact number."""

    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeVLLMGeneration:
    """The two seams `StepPhaseTimer.attach` wraps on TRL's colocated engine handle."""

    def __init__(self, clock: PhaseClock) -> None:
        self.clock = clock

    def sync_weights(self) -> None:
        self.clock.advance(1.0)

    def generate(
        self, *, prompts: object, images: object, num_generations: int
    ) -> tuple[object, ...]:
        del prompts, images, num_generations
        self.clock.advance(20.0)
        return ([[1]], [[2]], [[[0.0]]], [[[2]]])


class FakeTrainer:
    """A GRPOTrainer stand-in exposing exactly the seams the timer wraps, each costing known seconds.

    `_generate_and_score_completions` calls the inner seams the way TRL's does, so the test can
    check that nested phases are attributed once: the rollout total includes them, the unattributed
    remainder subtracts only the top-level phases.
    """

    def __init__(self, clock: PhaseClock, *, use_vllm: bool = True) -> None:
        self.clock = clock
        self.use_vllm = use_vllm
        self.vllm_generation = FakeVLLMGeneration(clock)
        self.processing_class = types.SimpleNamespace(batch_decode=self._batch_decode)
        self.accelerator = types.SimpleNamespace(backward=self._backward)
        self.model = types.SimpleNamespace(training=True)
        # Already the cross-process total: GRPOConfig folds num_processes into it (grpo_config.py:1085).
        self.args = types.SimpleNamespace(generation_batch_size=8)
        self._metrics: dict[str, dict[str, list[float]]] = {
            "train": collections.defaultdict(list),
            "eval": collections.defaultdict(list),
        }

    def _batch_decode(self, ids: object, skip_special_tokens: bool = True) -> list[str]:
        del ids, skip_special_tokens
        self.clock.advance(0.5)
        return ["decoded"]

    def _backward(self, loss: object) -> None:
        del loss
        self.clock.advance(3.0)

    def _generate_and_score_completions(self, inputs: object) -> dict[str, object]:
        del inputs
        self.vllm_generation.sync_weights()
        self.vllm_generation.generate(prompts=[[1]], images=None, num_generations=2)
        self.processing_class.batch_decode([[2]])
        self.processing_class.batch_decode([[2]])
        self._get_per_token_logps_and_entropies()
        self._calculate_rewards()
        self.clock.advance(2.0)  # tokenisation, gathers, advantages: rollout time outside the seams
        mode = "train" if self.model.training else "eval"  # as TRL's `_generate` picks its list
        self._metrics[mode]["completions/mean_length"].append(250.0)
        return {}

    def _calculate_rewards(self) -> None:
        self.clock.advance(4.0)

    def _get_per_token_logps_and_entropies(self) -> None:
        self.clock.advance(6.0)

    def compute_loss(
        self, model: object, inputs: object, num_items_in_batch: object = None
    ) -> float:
        del model, inputs, num_items_in_batch
        self.clock.advance(5.0)
        return 0.0


def drive_one_step(
    timer: tp.StepPhaseTimer, trainer: FakeTrainer, clock: PhaseClock, *, micro_steps: int = 2
) -> None:
    """Run the callback and seam sequence transformers' inner loop produces for one optimizer step."""
    args = types.SimpleNamespace()
    state = types.SimpleNamespace(global_step=1)
    control = types.SimpleNamespace()
    trainer._generate_and_score_completions([])
    for _ in range(micro_steps):
        trainer.compute_loss(None, {})
        trainer.accelerator.backward(0.0)
    clock.advance(0.75)  # the nan check, token tracking and gradient clipping before the optimizer
    timer.on_pre_optimizer_step(args, state, control)  # pyright: ignore[reportArgumentType]
    clock.advance(1.5)  # optimizer.step
    timer.on_optimizer_step(args, state, control)  # pyright: ignore[reportArgumentType]
    clock.advance(0.25)  # scheduler step, zero_grad, bookkeeping before on_step_end
    timer.on_step_end(args, state, control)  # pyright: ignore[reportArgumentType]


class TestStepPhaseTimer:
    """Per-phase attribution on a fake step whose every phase costs a known number of seconds.

    Sabotage recorded in the group report for this change: with the `generate` wrap removed from
    `attach`, `test_every_phase_is_attributed_exactly_once` reads `timing/generate_s` as 0.0 and
    the rollout remainder swallows the 20 s, so the assertion goes red.
    """

    @pytest.fixture
    def rig(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[tp.StepPhaseTimer, FakeTrainer, PhaseClock]:
        clock = PhaseClock()
        monkeypatch.setattr(tp.time, "perf_counter", clock.perf_counter)
        monkeypatch.setattr(tp.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(tp.torch.cuda, "reset_peak_memory_stats", lambda *_: None)
        timer = tp.StepPhaseTimer()
        monkeypatch.setattr(
            timer,
            "_memory_record",
            lambda: {"peak_allocated_gib": 1.0, "peak_reserved_gib": 2.0, "device_used_gib": 3.0},
        )
        trainer = FakeTrainer(clock)
        timer.attach(trainer)  # pyright: ignore[reportArgumentType]
        timer.on_train_begin(
            types.SimpleNamespace(),  # pyright: ignore[reportArgumentType]
            types.SimpleNamespace(global_step=0),  # pyright: ignore[reportArgumentType]
            types.SimpleNamespace(),  # pyright: ignore[reportArgumentType]
        )
        return timer, trainer, clock

    def test_every_phase_is_attributed_exactly_once(
        self, rig: tuple[tp.StepPhaseTimer, FakeTrainer, PhaseClock]
    ) -> None:
        timer, trainer, clock = rig
        drive_one_step(timer, trainer, clock)
        metrics = trainer._metrics["train"]
        seconds = {phase: metrics[tp.timing_metric(phase)] for phase in tp.STEP_PHASES}
        assert seconds["sync_weights"] == [1.0]
        assert seconds["generate"] == [20.0]
        assert seconds["decode"] == [1.0]
        assert seconds["rewards"] == [4.0]
        assert seconds["logps_forward"] == [6.0]
        # The whole rollout pass: its five inner seams plus the 2 s outside them.
        assert seconds["rollout_score"] == [1.0 + 20.0 + 1.0 + 4.0 + 6.0 + 2.0]
        assert seconds["compute_loss"] == [2 * 5.0]
        assert seconds["backward"] == [2 * 3.0]
        assert seconds["grad_clip"] == [0.75]
        assert seconds["optimizer_step"] == [1.5]
        interval = metrics[tp.STEP_INTERVAL_METRIC][0]
        assert interval == pytest.approx(34.0 + 10.0 + 6.0 + 0.75 + 1.5 + 0.25)
        # Only the top-level phases are subtracted: nested seams are already inside rollout_score.
        assert metrics[tp.UNATTRIBUTED_METRIC] == [pytest.approx(0.25)]
        assert metrics[tp.GENERATE_TOKENS_PER_SECOND_METRIC] == [pytest.approx(250.0 * 8 / 20.0)]
        assert all(value >= 0.0 for values in seconds.values() for value in values)

    def test_every_advertised_key_lands_in_the_record(
        self, rig: tuple[tp.StepPhaseTimer, FakeTrainer, PhaseClock]
    ) -> None:
        """`TIMING_METRIC_KEYS` is what mem_log.csv lays its columns out from; a key the timer
        writes but the tuple omits would be a column nobody gets, and the reverse an always-empty one."""
        timer, trainer, clock = rig
        drive_one_step(timer, trainer, clock)
        written = {
            key for key in trainer._metrics["train"] if key.startswith(tp.TIMING_METRIC_PREFIX)
        }
        assert written == set(tp.TIMING_METRIC_KEYS)

    def test_the_second_step_starts_from_zero(
        self, rig: tuple[tp.StepPhaseTimer, FakeTrainer, PhaseClock]
    ) -> None:
        timer, trainer, clock = rig
        drive_one_step(timer, trainer, clock)
        trainer._metrics["train"].clear()  # what GRPOTrainer.log does after each record
        drive_one_step(timer, trainer, clock, micro_steps=1)
        metrics = trainer._metrics["train"]
        assert metrics[tp.timing_metric("compute_loss")] == [5.0]
        assert metrics[tp.timing_metric("backward")] == [3.0]
        assert metrics[tp.timing_metric("generate")] == [20.0]

    def test_the_token_rate_counts_only_this_steps_completions_across_a_log_window(
        self, rig: tuple[tp.StepPhaseTimer, FakeTrainer, PhaseClock]
    ) -> None:
        """`GRPOTrainer.log` clears `_metrics` only when it logs, so under `--logging-steps N > 1`
        the second step of a window still sees the first step's `completions/mean_length` entry.
        Summing every entry there charged two steps of tokens to one step's generate seconds (the
        rate read 2x on step two, 3x on step three); each step's rate must be its own."""
        timer, trainer, clock = rig
        drive_one_step(timer, trainer, clock)
        drive_one_step(timer, trainer, clock)  # no clear in between: the window spans both steps
        metrics = trainer._metrics["train"]
        assert metrics["completions/mean_length"] == [250.0, 250.0]
        assert metrics[tp.GENERATE_TOKENS_PER_SECOND_METRIC] == [
            pytest.approx(250.0 * 8 / 20.0),
            pytest.approx(250.0 * 8 / 20.0),
        ]

    def test_an_eval_rollout_is_not_banked_against_the_training_rate(
        self, rig: tuple[tp.StepPhaseTimer, FakeTrainer, PhaseClock]
    ) -> None:
        """An eval-mode rollout appends to `_metrics["eval"]`, so reading the train list for it would
        raise on an empty list or re-count the previous training rollout. Its seconds still fall
        into the phase window they happen in (phases reset at `on_step_end` only), so the check is
        on the numerator: rate times generate seconds is one training rollout's tokens."""
        timer, trainer, clock = rig
        trainer.model.training = False
        trainer._generate_and_score_completions([])
        assert timer._completion_tokens == 0.0  # pyright: ignore[reportPrivateUsage]
        trainer.model.training = True
        drive_one_step(timer, trainer, clock)
        metrics = trainer._metrics["train"]
        assert metrics["completions/mean_length"] == [250.0]
        banked = (
            metrics[tp.GENERATE_TOKENS_PER_SECOND_METRIC][0]
            * metrics[tp.timing_metric("generate")][0]
        )
        assert banked == pytest.approx(250.0 * 8)

    def test_without_vllm_the_generation_seams_are_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On the transformers.generate path there is no engine handle to wrap; the rollout total
        still carries the generation time, `generate_s` reads zero and the token rate is NaN."""
        clock = PhaseClock()
        monkeypatch.setattr(tp.time, "perf_counter", clock.perf_counter)
        monkeypatch.setattr(tp.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(tp.torch.cuda, "reset_peak_memory_stats", lambda *_: None)
        timer = tp.StepPhaseTimer()
        monkeypatch.setattr(
            timer,
            "_memory_record",
            lambda: {"peak_allocated_gib": 1.0, "peak_reserved_gib": 2.0, "device_used_gib": 3.0},
        )
        trainer = FakeTrainer(clock, use_vllm=False)
        timer.attach(trainer)  # pyright: ignore[reportArgumentType]
        # A wrapped seam becomes an instance attribute; an untouched one stays a class method.
        assert "generate" not in vars(trainer.vllm_generation)
        assert "sync_weights" not in vars(trainer.vllm_generation)
        timer.on_train_begin(
            types.SimpleNamespace(),  # pyright: ignore[reportArgumentType]
            types.SimpleNamespace(global_step=0),  # pyright: ignore[reportArgumentType]
            types.SimpleNamespace(),  # pyright: ignore[reportArgumentType]
        )
        drive_one_step(timer, trainer, clock)
        metrics = trainer._metrics["train"]
        assert metrics[tp.timing_metric("generate")] == [0.0]
        assert metrics[tp.timing_metric("rollout_score")] == [34.0]
        assert math.isnan(metrics[tp.GENERATE_TOKENS_PER_SECOND_METRIC][0])

    def test_attaching_twice_is_refused(
        self, rig: tuple[tp.StepPhaseTimer, FakeTrainer, PhaseClock]
    ) -> None:
        """A second attach would wrap the wrappers and count every phase twice."""
        timer, trainer, _ = rig
        with pytest.raises(RuntimeError, match="already attached"):
            timer.attach(trainer)  # pyright: ignore[reportArgumentType]


def test_reward_varies_across_completions():
    """A constant reward would make frac_reward_zero_std uninformative."""
    rewards = tp.reward_token_variety(["aaaa", "abcd", "abab"])
    assert len(set(rewards)) > 1
    assert all(0.0 < r <= 1.0 for r in rewards)


def test_format_table_renders_a_result_without_a_gpu():
    result = {
        "config": {
            "model_id": "Qwen/Qwen3.5-4B",
            "prompts_per_step": 3,
            "group_size": 8,
            "prompt_tokens": 2048,
            "completion_tokens": 2048,
            "micro_batch_size": None,
            "use_liger_kernel": True,
            "gradient_checkpointing": True,
        },
        "device": {"device_name": "NVIDIA L4", "total_vram_gib": 22.06},
        "metrics": {
            "completions_mean_length": 2048.0,
            "reward": 0.2,
            "reward_std": 0.05,
            "frac_reward_zero_std": 0.0,
        },
        "memory_prediction": {"predicted_kv_gib": 3.0, "predicted_recurrent_gib": 1.2},
        "lora": {"trainable_params": 32464896, "trainable_fraction": 0.0071},
        "episodes_per_step": 24,
        "median_step_seconds": 500.0,
        "per_step_seconds_measured": [499.0, 501.0],
        "generation_fraction_of_step": 0.8,
        "episodes_per_hour": 172.8,
        "steps_per_hour": 7.2,
        "hours_per_200_steps": 27.8,
        "episode_tokens_per_second": 196.6,
        "generated_tokens_per_second": 98.3,
        "peak_allocated_gib": 20.0,
        "peak_reserved_gib": 20.5,
        "peak_device_used_gib": 21.0,
    }
    table = tp.format_table(result)  # pyright: ignore[reportArgumentType]
    assert "NVIDIA L4" in table
    assert "3 x 8" in table
    assert "8 x 3" in table, "micro-batch x accumulation should read 8 x 3 at P=3, G=8"


class TestSweepFailureClassification:
    """Naming the phase that ran out of memory, since the two failures have different fixes."""

    def test_generation_oom_is_named_as_such(self):
        traceback = (
            "  File .../trl/trainer/grpo_trainer.py, line 2242, in _generate\n"
            "  File .../grpo_trainer.py, line 1902, in _generate_single_turn\n"
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        )
        assert throughput_sweep.classify_failure(traceback) == "oom_generation"

    def test_training_pass_oom_is_named_as_such(self):
        traceback = (
            "  File .../peft/tuners/lora/layer.py, line 700, in forward\n"
            "    result = result + lora_B(lora_A(dropout(x))) * scaling\n"
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.12 GiB\n"
        )
        assert throughput_sweep.classify_failure(traceback) == "oom_training"

    def test_a_non_memory_failure_is_not_reported_as_an_oom(self):
        assert throughput_sweep.classify_failure("ValueError: something else") == "failed"
