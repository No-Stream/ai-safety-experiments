from __future__ import annotations

import csv
import os
import types
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest
from transformers import TrainerControl

from grpo import rlvr_math as rm
from grpo.estimator_defaults import GRPO_LOSS_TYPE, GRPO_SCALE_REWARDS

if TYPE_CHECKING:
    from pathlib import Path


def test_make_task_pairs_deterministic_and_seeded():
    task = rm.TaskConfig(
        task_mode="ltr", ltr_min_steps=2, ltr_max_steps=3, val_range=10, mul_range=3
    )
    train_seed = 11
    eval_seed = 99
    pairs_a_train, pairs_a_eval = rm._make_task_pairs(  # pyright: ignore[reportPrivateUsage]
        task, n_train=5, n_eval=4, train_seed=train_seed, eval_seed=eval_seed
    )
    pairs_b_train, pairs_b_eval = rm._make_task_pairs(  # pyright: ignore[reportPrivateUsage]
        task, n_train=5, n_eval=4, train_seed=train_seed, eval_seed=eval_seed
    )

    # Deterministic for same seeds/config
    assert pairs_a_train == pairs_b_train
    assert pairs_a_eval == pairs_b_eval

    # Train/eval should differ because seeds differ
    assert pairs_a_train != pairs_a_eval

    # Changing eval seed should change eval set but not train set
    _, pairs_c_eval = rm._make_task_pairs(  # pyright: ignore[reportPrivateUsage]
        task, n_train=5, n_eval=4, train_seed=train_seed, eval_seed=eval_seed + 1
    )
    assert pairs_c_eval != pairs_a_eval
    assert pairs_a_train == pairs_b_train


def test_parse_and_reward_correct_integer_handles_last_int_and_sign():
    texts = [
        "answer: 3",
        "nums -1 then 4",
        "no ints here",
        "foo -7 bar 5",
        "1,234 is the final value",
        "leading 10_000 then -9",
    ]
    gold = [3, 4, 0, 5, 1234, -9]
    # parse_answer uses last integer-like token
    assert rm.parse_answer(texts[0]) == 3
    assert rm.parse_answer(texts[1]) == 4
    assert rm.parse_answer(texts[2]) is None
    assert rm.parse_answer(texts[3]) == 5
    assert rm.parse_answer(texts[4]) == 1234
    assert rm.parse_answer(texts[5]) == -9

    rewards = rm.reward_correct_integer(texts, gold)
    assert rewards == [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]


class DummyIDs(list[int]):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, 2)


class DummyBatch(dict[str, object]):
    def to(self, device: object) -> DummyBatch:
        return self


class DummyTok:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> str:
        return "prompt"

    def __call__(self, text: str, return_tensors: str = "pt") -> DummyBatch:
        return DummyBatch({"input_ids": DummyIDs([0, 1])})

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "0"


def install_baseline_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub model/tokenizer to avoid heavy downloads, and record what the call site passed."""
    captured: dict[str, Any] = {}

    class DummyModel:
        config = types.SimpleNamespace(
            pad_token_id=None, eos_token_id=None, bos_token_id=None, generation_config=None
        )

        def eval(self) -> DummyModel:
            return self

        def generate(self, **kwargs: Any) -> list[list[int]]:
            captured["generate_kwargs"] = kwargs
            return [[0, 1, 2]]

    def fake_from_pretrained(*args: Any, **kwargs: Any) -> DummyModel:
        captured["from_pretrained_kwargs"] = kwargs
        return DummyModel()

    monkeypatch.setattr(
        rm, "AutoModelForCausalLM", types.SimpleNamespace(from_pretrained=fake_from_pretrained)
    )
    monkeypatch.setattr(
        rm, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTok())
    )
    return captured


def test_measure_baseline_uses_task_and_eval_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    install_baseline_stubs(monkeypatch)

    task_cfg = rm.TaskConfig(task_mode="simple")
    res = rm.measure_baseline_accuracy(
        model_id="dummy",
        task_cfg=task_cfg,
        n_eval=7,
        device="cpu",
        dtype=None,
        load_in_4bit=False,
        eval_seed=321,
    )
    # Should report the requested eval size and task mode
    assert res["n"] == 7
    assert res["task_mode"] == "simple"


def test_baseline_generation_passes_no_sampling_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Greedy decoding must not carry temperature/top_p, which transformers silently drops.

    The warpers are only built inside `if generation_config.do_sample`, so passing them under
    `do_sample=False` reads as "same sampler as training" while changing no number at all.
    """
    captured = install_baseline_stubs(monkeypatch)
    rm.measure_baseline_accuracy(
        model_id="dummy",
        task_cfg=rm.TaskConfig(task_mode="simple"),
        n_eval=2,
        device="cpu",
        dtype=None,
        load_in_4bit=False,
    )
    generate_kwargs = captured["generate_kwargs"]
    assert generate_kwargs["do_sample"] is False
    assert "temperature" not in generate_kwargs
    assert "top_p" not in generate_kwargs


def test_baseline_defaults_match_the_training_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """A baseline loaded at a different precision than the policy is not a comparison."""
    captured = install_baseline_stubs(monkeypatch)
    rm.measure_baseline_accuracy(
        model_id="dummy",
        task_cfg=rm.TaskConfig(task_mode="simple"),
        n_eval=2,
        device="cpu",
    )
    load_kwargs = captured["from_pretrained_kwargs"]
    assert load_kwargs["dtype"] == rm.TrainConfig.dtype
    assert "quantization_config" not in load_kwargs, (
        "4-bit is not the training default and quantizes this family badly"
    )


class StubGRPOConfig:
    """Accepts whatever GRPOConfig fields the call site sets, so a rename shows up in the test."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.world_size = getattr(self, "world_size", 1)
        self.steps_per_generation = getattr(self, "steps_per_generation", 1)
        self.generation_batch_size = getattr(self, "generation_batch_size", None)


class TrainTok:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.padding_side = "left"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        return "prompt"

    def __call__(
        self, text: str, return_tensors: str | None = None, add_special_tokens: bool = True
    ) -> DummyBatch:
        return DummyBatch({"input_ids": [0, 1]})

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "0"


def install_train_stubs(
    monkeypatch: pytest.MonkeyPatch, lora_target_modules: list[str]
) -> dict[str, Any]:
    """Stub GRPOTrainer/GRPOConfig/tokenizer to inspect what the call site builds, no training.

    `discover_lora_targets` is stubbed rather than called because the real one downloads a
    config from the hub, and `make test` stays offline.
    """
    captured: dict[str, Any] = {}

    class StubTrainer:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.args = kwargs["args"]
            self.model = types.SimpleNamespace(
                config=types.SimpleNamespace(), generation_config=types.SimpleNamespace()
            )

        def train(self) -> None:
            return

        def save_model(self) -> None:
            return

        def save_state(self) -> None:
            return

    monkeypatch.setattr(rm, "GRPOTrainer", StubTrainer)
    monkeypatch.setattr(rm, "GRPOConfig", StubGRPOConfig)
    monkeypatch.setattr(
        rm, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: TrainTok())
    )
    monkeypatch.setattr(
        rm,
        "discover_lora_targets",
        lambda model_id: {"target_modules": list(lora_target_modules)},
    )
    return captured


QWEN3_DENSE_TARGETS = [
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
]
QWEN35_HYBRID_TARGETS = [
    *QWEN3_DENSE_TARGETS,
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
]


def test_a_liger_unfaithful_loss_constant_refuses_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness hardcodes GRPO_LOSS_TYPE, so the silent path back to executed-GRPO-labelled-dapo
    is someone editing the constant; the launch must refuse rather than train under the old bug."""
    install_train_stubs(monkeypatch, QWEN3_DENSE_TARGETS)
    monkeypatch.setattr(rm, "GRPO_LOSS_TYPE", "dapo")
    with pytest.raises(ValueError, match="num_items_in_batch"):
        rm.train_grpo_integer_math(rm.TrainConfig(task=rm.TaskConfig(task_mode="simple")))


def test_the_acknowledgement_field_unblocks_the_unfaithful_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = install_train_stubs(monkeypatch, QWEN3_DENSE_TARGETS)
    monkeypatch.setattr(rm, "GRPO_LOSS_TYPE", "dapo")
    rm.train_grpo_integer_math(
        rm.TrainConfig(
            acknowledge_liger_estimator_mismatch=True, task=rm.TaskConfig(task_mode="simple")
        )
    )
    assert captured["args"].loss_type == "dapo"


def test_train_grpo_forwards_config_to_trainer_args(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = install_train_stubs(monkeypatch, QWEN3_DENSE_TARGETS)

    cfg = rm.TrainConfig(
        quick_run=True,
        max_steps_quick=123,
        train_seed=7,
        eval_seed=8,
        dtype=None,  # pyright: ignore[reportArgumentType]
        task=rm.TaskConfig(task_mode="simple"),
    )
    rm.train_grpo_integer_math(cfg)

    args = captured["args"]
    assert args.max_steps == 123
    assert args.seed == 7
    # The one knob whose value is a term of the loss rather than a size, so a literal at the call
    # site would leave it out of every record this config lands in.
    assert args.beta == cfg.beta
    assert args.learning_rate == cfg.learning_rate
    # The value as well as the round trip. At any nonzero beta TRL builds a reference model and adds
    # `beta * kl` to the loss (grpo_trainer.py:967-969, :3190-3191), which is a term no arm in
    # games/train.py has, and under dr_grpo with unscaled advantages it is the WHOLE gradient of a
    # step whose groups are pure. The compute-matched control inherits this config, so a nonzero value
    # here also puts the control on a different objective from the arm it controls for.
    assert args.beta == 0.0
    # The learning rate's value too, for the same control: it is the arms' 1e-5
    # (`games.train.GameTrainConfig`), and a control trained hotter than the arm it controls for is on
    # a different optimizer at the same step count, which no artifact downstream could recover.
    assert args.learning_rate == 1e-5
    # warmup is passed as a ratio under warmup_steps; warmup_ratio no longer exists upstream.
    assert args.warmup_steps == cfg.warmup_ratio
    assert not hasattr(args, "warmup_ratio")
    assert not hasattr(args, "max_prompt_length")
    # TRL reads `dtype` and ignores `torch_dtype`, which is what silently forced float32 before.
    assert "dtype" in args.model_init_kwargs
    assert "torch_dtype" not in args.model_init_kwargs


def test_lora_targets_come_from_the_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hardcoded target list leaves every Gated DeltaNet token mixer frozen on Qwen3.5.

    Three quarters of that stack is linear attention, whose projections are named nothing like
    `q_proj`, and the run reports a healthy trainable-parameter count either way.
    """
    captured = install_train_stubs(monkeypatch, QWEN35_HYBRID_TARGETS)
    rm.train_grpo_integer_math(rm.TrainConfig(task=rm.TaskConfig(task_mode="simple")))

    targets = captured["peft_config"].target_modules
    assert "in_proj_qkv" in targets
    assert "out_proj" in targets
    assert "q_proj" in targets
    assert "lm_head" not in targets, (
        "TRL refuses to start with a head adapter under use_liger_kernel"
    )


def test_grpo_config_accepts_the_arguments_we_pass(tmp_path: Path) -> None:
    """Guard against upstream removing a GRPOConfig field we depend on.

    The repo was previously broken by exactly this: TRL dropped `max_prompt_length` and
    `warmup_ratio`, and the failure only surfaced at training time. This builds a real
    GRPOConfig, so a rename or removal fails fast and cheaply on CPU.
    """
    cfg = rm.GRPOConfig(
        output_dir=str(tmp_path),
        warmup_steps=0.1,
        max_completion_length=128,
        num_generations=8,
        per_device_train_batch_size=8,
        beta=rm.TrainConfig.beta,
        epsilon=0.2,
        scale_rewards=GRPO_SCALE_REWARDS,
        loss_type=GRPO_LOSS_TYPE,
        use_liger_kernel=True,
        temperature=0.8,
        top_p=0.9,
        tf32=True,
        bf16=True,
        report_to="none",
    )
    assert cfg.scale_rewards == GRPO_SCALE_REWARDS
    assert cfg.loss_type == GRPO_LOSS_TYPE
    # A float below 1 must be read as a fraction of total steps, not as 0 literal steps.
    assert cfg.get_warmup_steps(200) == 20


@pytest.mark.skipif(not os.getenv("RLVR_SMOKE"), reason="RLVR_SMOKE not set")
def test_baseline_smoke_tiny_gpt2():
    """End-to-end CPU pass through the real transformers stack on a ~2 MB model.

    A load failure here must fail rather than skip: this is the only test that exercises the
    installed transformers/torch against real weights, so swallowing the error would let a
    broken environment report green.
    """
    model_id = "sshleifer/tiny-gpt2"
    res = rm.measure_baseline_accuracy(
        model_id=model_id,
        task_cfg=rm.TaskConfig(task_mode="simple", val_range=10, mul_range=3),
        n_eval=3,
        device="cpu",
        dtype=rm.torch.float32,
        load_in_4bit=False,
        eval_seed=5,
        chat_template="You are a calculator.\nUser: {{user}}\nAssistant:",
    )
    assert res["n"] == 3


class FakeClock:
    """Deterministic wall clock, so the EMA step time comes out an exact number."""

    def __init__(self, step_seconds: float = 1.0) -> None:
        self.now = 1_000.0
        self.step_seconds = step_seconds

    def time(self) -> float:
        return self.now

    def advance(self) -> None:
        self.now += self.step_seconds


FIXED_MEM_STATS = {
    "free_gib": 6.0,
    "total_gib": 24.0,
    "used_gib": 18.0,
    "alloc_gib": 12.0,
    "reserved_gib": 14.0,
}


class TestMemoryMonitorCallback:
    """The telemetry that decides whether a configuration is worth a Batch run."""

    def _run_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        logs_per_step: list[dict[str, float]],
        *,
        extra_columns: tuple[str, ...] = (),
    ) -> list[dict[str, str]]:
        clock = FakeClock()
        monkeypatch.setattr(rm, "time", clock)
        callback = rm.MemoryMonitorCallback(extra_columns=extra_columns)
        monkeypatch.setattr(callback, "_gpu_mem_stats", lambda: dict(FIXED_MEM_STATS))
        args = types.SimpleNamespace(
            output_dir=str(tmp_path),
            per_device_train_batch_size=16,
            world_size=1,
            gradient_accumulation_steps=2,
            num_generations=8,
            max_completion_length=128,
            max_steps=10,
        )
        state = types.SimpleNamespace(global_step=0, is_world_process_zero=True)
        control = TrainerControl()
        callback.on_train_begin(args, state, control)  # pyright: ignore[reportArgumentType]
        for step, logs in enumerate(logs_per_step, start=1):
            state.global_step = step
            clock.advance()
            callback.on_log(args, state, control, logs=logs)  # pyright: ignore[reportArgumentType]
        with (tmp_path / "mem_log.csv").open() as f:
            return list(csv.DictReader(f))

    def _run_two_steps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logs: dict[str, float]
    ) -> list[dict[str, str]]:
        return self._run_steps(monkeypatch, tmp_path, [logs, logs])

    def test_completions_per_second_counts_each_completion_once_per_optimizer_step(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """TRL's per-device batch already counts completions, and a step is an optimizer step.

        Multiplying by num_generations double-counts the group dimension, and omitting
        gradient accumulation undercounts the completions a single global_step consumed.
        """
        rows = self._run_two_steps(monkeypatch, tmp_path, {})
        last = rows[-1]
        assert float(last["ema_step_s"]) == pytest.approx(1.0)
        # 16 completions x 1 device x 2 accumulation steps = 32 completions per optimizer step.
        assert float(last["approx_seq_s"]) == pytest.approx(32.0)

    def test_tokens_per_second_is_the_real_num_tokens_delta_over_the_log_interval(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Real tokens, never completions times `max_completion_length`.

        The padded figure read `tok/s~2867` on a run whose generation actually ran at about 450
        tokens/s (efficiency audit 2026-09-02): 32 completions x 128 budget here would report
        4096, while the run below generated 3000 tokens in a one-second step. The first log record
        has no earlier `num_tokens` to difference against and writes an empty cell, not a guess.
        """
        rows = self._run_steps(
            monkeypatch,
            tmp_path,
            [{"num_tokens": 10_000.0}, {"num_tokens": 13_000.0}, {"num_tokens": 13_500.0}],
        )
        assert rows[0]["tok_s"] == ""
        assert float(rows[1]["tok_s"]) == pytest.approx(3000.0)
        assert float(rows[2]["tok_s"]) == pytest.approx(500.0)
        assert "approx_tok_s" not in rows[0]

    def test_a_record_without_num_tokens_leaves_the_rate_blank_and_the_baseline_intact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A bare eval record between two training records must not zero the next rate."""
        rows = self._run_steps(
            monkeypatch,
            tmp_path,
            [{"num_tokens": 10_000.0}, {"quick_eval_accuracy": 0.5}, {"num_tokens": 12_000.0}],
        )
        assert rows[1]["tok_s"] == ""
        # Two fake-clock seconds elapsed since the last record that carried num_tokens.
        assert float(rows[2]["tok_s"]) == pytest.approx(1000.0)

    def test_extra_columns_are_copied_from_the_log_record_and_blank_when_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The phase timer's seconds and the trim totals ride in the same row as the memory reading."""
        rows = self._run_steps(
            monkeypatch,
            tmp_path,
            [{"timing/generate_s": 12.25, "num_tokens": 1.0}, {"num_tokens": 2.0}],
            extra_columns=("timing/generate_s", "padding_trim/step_padded_tokens"),
        )
        assert list(rows[0])[-2:] == ["timing/generate_s", "padding_trim/step_padded_tokens"]
        assert float(rows[0]["timing/generate_s"]) == pytest.approx(12.25)
        assert rows[0]["padding_trim/step_padded_tokens"] == ""
        assert rows[1]["timing/generate_s"] == ""

    def test_csv_is_the_only_sink_and_the_logs_dict_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Trainer.log has already copied `logs` into log_history when a callback sees it.

        Writing telemetry back into that dict persists nothing, so the CSV must carry it all.
        """
        logs = {"reward": 0.5}
        rows = self._run_two_steps(monkeypatch, tmp_path, logs)
        assert logs == {"reward": 0.5}
        assert float(rows[-1]["used_gib"]) == pytest.approx(18.0)
        assert float(rows[-1]["reserved_gib"]) == pytest.approx(14.0)
        assert float(rows[-1]["usage_pct"]) == pytest.approx(75.0)

    def test_a_resumed_launch_appends_to_the_earlier_rows_instead_of_truncating(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The rows of the steps a killed run paid for must survive its restart.

        `on_train_begin` fires after the trainer state has been loaded from the checkpoint, so a
        resumed launch arrives with `global_step` past zero and an existing CSV; rewriting the header
        there threw away every earlier row (the behaviour before this test).
        """
        self._run_two_steps(monkeypatch, tmp_path, {"num_tokens": 1.0})
        callback = rm.MemoryMonitorCallback()
        monkeypatch.setattr(callback, "_gpu_mem_stats", lambda: dict(FIXED_MEM_STATS))
        args = types.SimpleNamespace(output_dir=str(tmp_path))
        resumed_state = types.SimpleNamespace(global_step=2, is_world_process_zero=True)
        callback.on_train_begin(args, resumed_state, TrainerControl())  # pyright: ignore[reportArgumentType]
        with (tmp_path / "mem_log.csv").open() as f:
            lines = f.read().splitlines()
        assert len(lines) == 3, "the header and both earlier rows must still be there"
        assert lines[0].startswith("time,step,")
        fresh_state = types.SimpleNamespace(global_step=0, is_world_process_zero=True)
        callback.on_train_begin(args, fresh_state, TrainerControl())  # pyright: ignore[reportArgumentType]
        with (tmp_path / "mem_log.csv").open() as f:
            assert len(f.read().splitlines()) == 1, "a fresh launch starts the CSV over"

    def test_a_resumed_launch_starts_its_step_clock_at_the_resumed_step(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The first record after a resume at step N is one step of wall clock, not N of them.

        With `_last_step` left at its initial zero, that record divided the elapsed second by N and
        every EMA and ETA after it inherited the error: the group's own L4 rehearsal read
        `ema_step_s=10.53` for a 21.06 s step resumed at step 1.
        """
        self._run_two_steps(monkeypatch, tmp_path, {"num_tokens": 1.0})
        clock = FakeClock()
        monkeypatch.setattr(rm, "time", clock)
        callback = rm.MemoryMonitorCallback()
        monkeypatch.setattr(callback, "_gpu_mem_stats", lambda: dict(FIXED_MEM_STATS))
        args = types.SimpleNamespace(
            output_dir=str(tmp_path),
            per_device_train_batch_size=16,
            world_size=1,
            gradient_accumulation_steps=2,
            max_steps=10,
        )
        state = types.SimpleNamespace(global_step=2, is_world_process_zero=True)
        callback.on_train_begin(args, state, TrainerControl())  # pyright: ignore[reportArgumentType]
        state.global_step = 3
        clock.advance()
        callback.on_log(args, state, TrainerControl(), logs={"num_tokens": 2.0})  # pyright: ignore[reportArgumentType]
        with (tmp_path / "mem_log.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert [row["step"] for row in rows] == ["1", "2", "3"]
        assert float(rows[-1]["ema_step_s"]) == pytest.approx(1.0)
        assert float(rows[-1]["approx_seq_s"]) == pytest.approx(32.0)

    def test_a_resumed_launch_rotates_a_csv_written_under_an_older_header(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Every run launched before `tok_s` and `extra_columns` wrote nine columns; appending
        twenty-three-field rows to that file made it ragged, and `load_mem_log`'s `read_csv` then
        raised inside `_summarize_run` -- after the training had finished, before the final sync.
        The old file is set aside with its rows intact, a fresh header is written, and the loader
        still sees the earlier segment (its peak is still the run's peak).
        """
        old_header = "time,step,used_gib,alloc_gib,reserved_gib,usage_pct,ema_step_s,approx_tok_s,approx_seq_s\n"
        old_rows = "1000,1,21.5,12.0,14.0,89.58,1.0,4096.0,32.0\n1001,2,18.0,12.0,14.0,75.0,1.0,4096.0,32.0\n"
        (tmp_path / "mem_log.csv").write_text(old_header + old_rows)
        clock = FakeClock()
        monkeypatch.setattr(rm, "time", clock)
        callback = rm.MemoryMonitorCallback(extra_columns=("timing/generate_s",))
        monkeypatch.setattr(callback, "_gpu_mem_stats", lambda: dict(FIXED_MEM_STATS))
        args = types.SimpleNamespace(
            output_dir=str(tmp_path),
            per_device_train_batch_size=16,
            world_size=1,
            gradient_accumulation_steps=2,
            max_steps=10,
        )
        state = types.SimpleNamespace(global_step=2, is_world_process_zero=True)
        callback.on_train_begin(args, state, TrainerControl())  # pyright: ignore[reportArgumentType]
        state.global_step = 3
        clock.advance()
        callback.on_log(
            args,  # pyright: ignore[reportArgumentType]
            state,  # pyright: ignore[reportArgumentType]
            TrainerControl(),
            logs={"num_tokens": 2.0, "timing/generate_s": 7.5},
        )
        rotated = sorted(tmp_path.glob("mem_log.*.csv"))
        assert len(rotated) == 1, rotated
        assert rotated[0].read_text() == old_header + old_rows
        with (tmp_path / "mem_log.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["step"] == "3"
        assert float(rows[0]["timing/generate_s"]) == pytest.approx(7.5)
        combined = rm.load_mem_log(str(tmp_path))
        assert combined is not None
        assert combined["step"].tolist() == [1, 2, 3]
        peak = float(cast("float", combined["used_gib"].max()))
        assert peak == pytest.approx(21.5), "the pre-resume peak survives"
        assert pd.isna(combined["timing/generate_s"].iloc[0])


class TestMemLogThroughputColumn:
    """`summarize_logs` and `plot_memory` read whichever throughput column a mem_log carries."""

    def test_the_real_rate_column_is_read_and_its_blank_cells_ignored(self) -> None:
        frame = pd.DataFrame(
            {
                "step": [1, 2, 3],
                "used_gib": [18.0, 19.0, 18.5],
                "tok_s": [float("nan"), 400.0, 500.0],
            }
        )
        assert rm.mem_log_throughput_column(frame) == "tok_s"
        out = rm.summarize_logs(None, frame)
        assert out["mean_tok_s"] == pytest.approx(450.0)
        assert out["peak_used_gib"] == pytest.approx(19.0)

    def test_a_legacy_file_still_yields_its_padded_estimate_with_zeros_dropped(self) -> None:
        frame = pd.DataFrame(
            {"step": [1, 2], "used_gib": [18.0, 18.0], "approx_tok_s": [0.0, 4096.0]}
        )
        assert rm.mem_log_throughput_column(frame) == "approx_tok_s"
        assert rm.summarize_logs(None, frame)["mean_tok_s"] == pytest.approx(4096.0)

    def test_a_frame_with_neither_column_reports_no_rate(self) -> None:
        frame = pd.DataFrame({"step": [1], "used_gib": [18.0]})
        assert rm.mem_log_throughput_column(frame) is None
        assert "mean_tok_s" not in rm.summarize_logs(None, frame)


def test_plot_memory_fails_loudly_on_a_frame_without_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every mem_log.csv the callback has ever written carries `usage_pct`.

    A fallback for a column name no writer produces silently drops the panel instead of
    telling us the CSV is malformed.
    """
    monkeypatch.setattr(rm.plt, "show", lambda: None)
    frame = pd.DataFrame({"step": [1], "used_gib": [18.0], "reserved_gib": [14.0]})
    with pytest.raises(KeyError):
        rm.plot_memory(frame)
    rm.plt.close("all")


class StubLoggingTrainer:
    """Mimics the one part of Trainer.log that matters here.

    `CallbackHandler.on_log` clears `control.should_log` before dispatching, so an out-of-band
    `trainer.log()` call inside another callback cancels the step's own scheduled log record.
    """

    def __init__(self, control: TrainerControl) -> None:
        self.control = control
        self.logged: list[dict[str, float]] = []

    def log(self, logs: dict[str, float]) -> None:
        self.logged.append(logs)
        self.control.should_log = False


def test_quick_eval_leaves_the_pending_log_record_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise an eval step loses its loss, grad_norm and learning_rate as well as reward.

    At TrainConfig defaults the final step is also an eval step, so `summarize_logs` would
    report the second-to-last step's loss as the final training loss.
    """
    callback = rm.QuickEvalCallback(
        DummyTok(),  # pyright: ignore[reportArgumentType]
        [("1 + 1 = ?", 2)],
        n_quick=1,
        device="cpu",
        every_n_steps=3,
    )
    monkeypatch.setattr(callback, "_accuracy", lambda: 0.5)
    control = TrainerControl(should_log=True)
    trainer = StubLoggingTrainer(control)
    callback.set_trainer(trainer)  # pyright: ignore[reportArgumentType]
    state = types.SimpleNamespace(global_step=3, is_world_process_zero=True)

    callback.on_step_end(None, state, control)  # pyright: ignore[reportArgumentType]

    assert trainer.logged == [{"quick_eval_accuracy": 0.5}]
    assert control.should_log, (
        "the scheduled log record for this step must still be written; the handler clears the "
        "flag on any out-of-band trainer.log call"
    )


class TestNonFiniteMetricCallback:
    """A NaN loss at step 3 would otherwise write every later checkpoint from dead weights.

    Nothing else on the training path catches it. `read_back_metrics` checks a metric is PRESENT and
    reports metrics that never varied, and a NaN series is neither -- `nan != nan`, so it reads as
    having varied. The save, the summary, the S3 sync and the end-of-run gate would all succeed on two
    arms' worth of paid GPU time, and every downstream reader would measure a NaN model and read it as
    an arm whose training did nothing.
    """

    def _log(self, logs: dict[str, float] | None, *, step: int = 3) -> None:
        callback = rm.NonFiniteMetricCallback()
        state = types.SimpleNamespace(global_step=step, is_world_process_zero=True)
        callback.on_log(
            types.SimpleNamespace(logging_steps=1),  # pyright: ignore[reportArgumentType]
            state,  # pyright: ignore[reportArgumentType]
            TrainerControl(),
            logs=logs,
        )

    @pytest.mark.parametrize("metric", ["loss", "reward", "grad_norm"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_watched_metric_stops_the_run(self, metric: str, bad: float) -> None:
        with pytest.raises(RuntimeError, match="is not finite") as caught:
            self._log({"loss": 0.5, "reward": 0.25, "grad_norm": 1.0, metric: bad})
        message = str(caught.value)
        assert metric in message
        assert "step 3" in message

    def test_an_ordinary_step_passes_untouched(self) -> None:
        self._log({"loss": 0.5, "reward": 0.25, "grad_norm": 1.0, "learning_rate": 1e-5})

    def test_a_zero_loss_is_finite_and_allowed(self) -> None:
        """Zero is a real value on this reward, so a truthiness test here would fail every clean run."""
        self._log({"loss": 0.0, "reward": 0.0, "grad_norm": 0.0})

    def test_a_record_missing_the_watched_metrics_is_not_an_error(self) -> None:
        """QuickEvalCallback logs a bare accuracy record, and TRL logs metrics on its own schedule."""
        self._log({"quick_eval_accuracy": 0.5})
        self._log(None)
        self._log({})

    def test_a_non_finite_metric_nobody_watches_is_left_alone(self) -> None:
        """The watched set is the three that decide whether the weights are still alive."""
        self._log({"loss": 0.5, "kl": float("nan")})

    def test_the_one_shared_build_callbacks_attaches_it(self) -> None:
        """One `build_callbacks` for both trainers, and it attaches the guard.

        Both halves matter. Sharing is what makes "wired into both threads" a property of one function
        rather than of two lists that can drift, and the reward-hacking trainer reaching the same
        object is what says the sharing took. Imported inside the test because both trainers pull in
        the whole training stack and this module is otherwise a cheap check on the substrate.

        The attachment is asserted on the RETURNED list, not on the function's source: a substring
        check stays green when the constructor call is still in the body but no longer reachable.
        """
        from games import train as games_train  # noqa: PLC0415
        from reward_hacking import train as rh_train  # noqa: PLC0415

        assert rh_train.build_callbacks is games_train.build_callbacks
        callbacks = games_train.build_callbacks(logging_steps=10, output_dir="unused", s3_dest="")
        assert any(isinstance(cb, rm.NonFiniteMetricCallback) for cb in callbacks)
