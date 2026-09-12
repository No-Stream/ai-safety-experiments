"""Run `train_game_arm` end to end offline, and pin the ordering its correctness rests on.

Everything that needs a GPU, a hub download or weights is stubbed: the device description, the
tokenizer, the architecture reads, and `GRPOTrainer` itself. Everything else is the real thing --
prompt generation, the dataset build, the sizing arithmetic, `GRPOConfig` construction, the reward
function, the metric read-back and every artifact written. That split is deliberate: the failures
this file exists to catch are *ordering* failures, and ordering is exactly what a suite of
per-function tests cannot see.

Three of those orderings have teeth, and each was a live defect when this file was written:

*   The resume checkpoint is resolved, and the recorded launch it belongs to checked, BEFORE the
    sizing plan is derived from live free VRAM and before `run_config.json` is written. Without
    that, another process holding two gibibytes on a shared card silently re-plans a resumed arm
    at a smaller group size -- the GRPO advantage baseline and the population the opponent mix is
    estimated from -- and overwrites the only record of what the earlier steps were trained under.
*   `repair_scalar_eos` runs before `GRPOTrainer` is constructed, because TRL builds its own
    `GenerationConfig` from the tokenizer at construction time and overrides the model's. A repair
    applied afterwards is applied to nothing.
*   A `vs-fixed-mix` arm's rows carry a real frozen-opponent probability before a model is loaded,
    rather than raising inside the reward function one full generation batch later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest
import torch

from games import preflight, sizing
from games import train as gt
from grpo import estimator_defaults

if TYPE_CHECKING:
    from collections.abc import Sequence

# Enough of a Qwen3.5-shaped text config for the sizing arithmetic, with the linear-attention
# fields the hybrid stack charges for.
HYBRID_CONFIG = SimpleNamespace(
    layer_types=["linear_attention"] * 24 + ["full_attention"] * 8,
    num_key_value_heads=4,
    head_dim=256,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
)

STUB_DEVICE: dict[str, object] = {
    "device_name": "NVIDIA L40S (stub)",
    "compute_capability": "8.9",
    "total_vram_gib": 44.4,
    "free_vram_gib_at_start": 44.0,
}

# What the stubbed attention probe reports; the real one needs a loaded model on a CUDA device.
STUB_ATTENTION: dict[str, object] = {
    "hf_attn_implementation": "sdpa (stub)",
    "sdpa_probe": {"causal_no_mask": "FLASH_ATTENTION", "boolean_mask": "EFFICIENT_ATTENTION"},
    "vllm": {"available": True, "kv_cache_groups": [[{"backend": "FLASH_ATTN (stub)"}]]},
}

# A prompt render that leaves `<think>` open, i.e. the Qwen3.5/3.8 prefilled-thinking case.
PREFILLED_RENDER_SUFFIX = "<|im_start|>assistant\n<think>\n"


class StubTokenizer:
    """The tokenizer surface `train_game_arm` touches, and nothing else.

    Callable because `games.dataset` counts a rendered prompt's tokens by calling the tokenizer.
    One token per four characters, so the over-length drop is reachable from a token budget.
    """

    def __init__(self, *, has_reasoning_effort: bool = False) -> None:
        self.pad_token_id: int | None = None
        self.eos_token_id = 151645
        self.eos_token = "<|im_end|>"
        self.padding_side = "right"
        self.has_reasoning_effort = has_reasoning_effort
        self.template_kwargs_seen: list[dict[str, Any]] = []

    def apply_chat_template(self, conversation: Sequence[dict[str, str]], **kwargs: Any) -> str:
        self.template_kwargs_seen.append(dict(kwargs))
        content = conversation[-1]["content"]
        return f"{content}{PREFILLED_RENDER_SUFFIX}"

    def get_chat_template(self) -> str:
        knob = "reasoning_effort" if self.has_reasoning_effort else "enable_thinking"
        return "{%- if " + knob + " %}...{%- endif %}"

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text) // 4))}


class StubModel:
    """A stand-in for the model TRL would have loaded: named modules, parameters, config objects."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(pad_token_id=None, eos_token_id=None)
        self.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=None)

    def named_modules(self) -> list[tuple[str, object]]:
        return [
            ("model.layers.0.linear_attn.in_proj_qkv.lora_A.default", object()),
            ("model.layers.3.self_attn.q_proj.lora_A.default", object()),
        ]

    def parameters(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(numel=lambda: 1_000, requires_grad=True),
            SimpleNamespace(numel=lambda: 99_000, requires_grad=False),
        ]


class StubTrainer:
    """A GRPOTrainer stand-in that records what it was built with and what train() was asked."""

    instances: list[StubTrainer] = []  # noqa: RUF012

    def __init__(self, **kwargs: Any) -> None:
        self.build_kwargs = kwargs
        self.args = kwargs["args"]
        self.model = StubModel()
        self.accelerator = SimpleNamespace(num_processes=1)
        self.state = SimpleNamespace(log_history=[], global_step=0)
        # Read off the arguments exactly as GRPOTrainer.__init__ does: this is the seam the
        # consumption-point vLLM check reads, so the stub must not hardcode either side of it.
        self.use_vllm = self.args.use_vllm
        self.vllm_mode = self.args.vllm_mode
        self.num_generations = self.args.num_generations
        self.temperature = self.args.temperature
        self.top_p = self.args.top_p
        self.top_k = self.args.top_k
        self.min_p = None
        self.repetition_penalty = 1.0
        self.max_completion_length = self.args.max_completion_length
        self.generation_config = SimpleNamespace(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=None,
            repetition_penalty=1.0,
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            cache_implementation=None,
        )
        self.resumed_from: object = "never called"
        self.saved_model = False
        self.saved_state = False
        StubTrainer.instances.append(self)

    # A test sabotages the trace by overriding which steps get a file and with how many rows.
    trace_steps: tuple[int, ...] = (1, 2)
    trace_rows: dict[int, int] = {}  # noqa: RUF012

    def train(self, *, resume_from_checkpoint: str | None = None) -> None:
        self.resumed_from = resume_from_checkpoint
        # One healthy log record per step, carrying every metric a games run claims to produce.
        self.state.log_history = [
            {
                "step": step,
                "reward": 0.4 + 0.01 * step,
                "reward_std": 0.1,
                "frac_groups_pure": 0.1 * step,
                "parse_failure_rate": 0.02,
                "truncated_thinking_rate": 0.01,
                "coop_rate": 0.3 + 0.01 * step,
                "mean_keep_fraction": 0.5,
            }
            for step in (1, 2)
        ]
        self.state.global_step = 2
        generation_batch_size = cast("int", self.args.generation_batch_size)
        for step in self.trace_steps:
            self.write_trace_file(step, self.trace_rows.get(step, generation_batch_size))

    def write_trace_file(self, step: int, rows: int) -> Path:
        """Write the per-step completions parquet the way `GRPOTrainer.log` does: a pandas frame."""
        path = preflight.trace_file_path(Path(self.args.output_dir), step)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "step": [step] * rows,
                "prompt": [f"prompt-{i}" for i in range(rows)],
                "completion": [f"completion-{i}" for i in range(rows)],
                "advantage": [0.0] * rows,
            }
        ).to_parquet(path)
        return path

    def save_model(self) -> None:
        self.saved_model = True

    def save_state(self) -> None:
        self.saved_state = True


@dataclass(frozen=True)
class RunHarness:
    """What a stubbed run exposes to a test: the trainer it built and where it wrote."""

    trainer: StubTrainer
    output_dir: Path

    @property
    def run_config(self) -> dict[str, Any]:
        return json.loads((self.output_dir / gt.RUN_CONFIG_FILENAME).read_text())

    @property
    def summary(self) -> dict[str, Any]:
        return json.loads((self.output_dir / gt.TRAIN_SUMMARY_FILENAME).read_text())


@pytest.fixture
def stubbed_run(monkeypatch: pytest.MonkeyPatch) -> StubTokenizer:
    """Replace every GPU, hub and weights touch in `train_game_arm` with an offline stand-in."""
    tokenizer = StubTokenizer()
    StubTrainer.instances.clear()
    monkeypatch.setattr(gt, "describe_device", lambda: dict(STUB_DEVICE))
    monkeypatch.setattr(
        preflight.AutoTokenizer, "from_pretrained", classmethod(lambda _cls, *_a, **_k: tokenizer)
    )
    monkeypatch.setattr(
        gt,
        "discover_lora_targets",
        lambda _model_id: {
            "target_modules": ["in_proj_qkv", "q_proj"],
            "module_counts": {"linear_attn": 24, "self_attn": 8},
            "expected_linear_attention_layers": 24,
        },
    )
    monkeypatch.setattr(gt, "log_deltanet_kernel_paths", lambda **_k: {"chunk": "fla (stub)"})
    monkeypatch.setattr(
        gt, "checkpoint_sequence_cost", lambda _model_id: sizing.sequence_cost(HYBRID_CONFIG)
    )
    monkeypatch.setattr(gt, "count_meta_parameters", lambda _model_id: 4_000_000_000)
    monkeypatch.setattr(gt, "DynamicSampledGRPOTrainer", StubTrainer)
    monkeypatch.setattr(gt, "peak_memory_gib", lambda _output_dir: {"peak_allocated_gib": 1.0})
    # The phase timer wraps seams only a real GRPOTrainer has, and the attention probe needs a
    # loaded model on a device; both are stubbed at their seam and asserted on separately.
    monkeypatch.setattr(gt.StepPhaseTimer, "attach", lambda _self, _trainer: None)
    monkeypatch.setattr(gt, "describe_attention_backends", lambda _trainer: dict(STUB_ATTENTION))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(
        preflight,
        "repair_scalar_eos",
        lambda _tokenizer, model_id: {"repaired": False, "reason": f"stub for {model_id}"},
    )
    return tokenizer


def arm_config(tmp_path: Path, **overrides: object) -> gt.GameTrainConfig:
    """The two-step, eight-prompt arm every stubbed run here trains, with `overrides` applied."""
    settings: dict[str, object] = {
        "arm": "twin-pd-group",
        "generate_fresh": True,
        "output_dir": str(tmp_path / "run"),
        "max_prompts": 8,
        "max_steps": 2,
        "num_generations": 4,
        "prompts_per_step": 2,
    } | overrides
    return gt.GameTrainConfig(**settings)  # pyright: ignore[reportArgumentType]


def run_arm(tmp_path: Path, **overrides: object) -> RunHarness:
    """Train one arm through the stubbed path and hand back its trainer and artifacts."""
    config = arm_config(tmp_path, **overrides)
    gt.train_game_arm(config)
    return RunHarness(
        trainer=StubTrainer.instances[-1], output_dir=Path(cast("str", config.output_dir))
    )


def seed_checkpoint(output_dir: Path, step: int, *, omit: tuple[str, ...] = ()) -> Path:
    """Lay down `checkpoint-<step>` with every file a resume needs, minus the ones named in `omit`.

    `trainer_state.json` carries the step, because that is what both the resume and the
    already-complete check read; every other file is empty, since the completeness gate reads
    names and nothing else.
    """
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / gt.TRAINER_STATE_FILENAME).write_text(json.dumps({"global_step": step}))
    for name in (*gt.REQUIRED_CHECKPOINT_FILES, gt.INIT_ADAPTER_WEIGHTS_FILENAME):
        if name not in omit and name != gt.TRAINER_STATE_FILENAME:
            if name == gt.ADAPTER_CONFIG_FILENAME:
                (checkpoint / name).write_text(json.dumps({"peft_type": "LORA"}))
            else:
                (checkpoint / name).write_bytes(b"state")
    return checkpoint


def snapshot_tree(root: Path) -> dict[str, bytes]:
    """Every file under `root` with its bytes, so a test can assert that nothing at all changed."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def free_vram_leaving_room_for(recorded_plan: dict[str, Any], *, episodes: float) -> float:
    """The free-VRAM reading that would leave headroom for exactly `episodes` episodes.

    Inverted from the plan a launch already recorded rather than rebuilt from the sizing model's
    constants, so the sabotage stays valid when the overhead figure or the per-episode cost is
    re-measured. `usable_vram_gib - episode_headroom_gib` is the part that does not scale with the
    batch: the weights plus the measured unmodelled overhead. The resident engine's share comes
    back on top because the plan subtracts it from the raw reading before the usable fraction.
    """
    fixed_gib = recorded_plan["usable_vram_gib"] - recorded_plan["episode_headroom_gib"]
    usable = fixed_gib + episodes * recorded_plan["predicted_episode_gib"]
    return usable / sizing.DEFAULT_VRAM_USABLE_FRACTION + recorded_plan["engine_reserved_gib"]


class TestTrainGameArmWiring:
    """The assembly, not the pieces: what the run built, in what order, and what it recorded."""

    def test_a_run_trains_saves_and_writes_both_artifacts(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        harness = run_arm(tmp_path)
        assert harness.trainer.saved_model is True
        assert harness.trainer.saved_state is True
        assert harness.run_config["arm"] == "twin-pd-group"
        assert harness.summary["steps_completed"] == 2
        assert harness.summary["missing_metrics"] == []

    def test_the_sizing_plan_it_derived_is_what_reached_trl_and_the_reward(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        harness = run_arm(tmp_path)
        plan = harness.run_config["sizing_plan"]
        assert harness.trainer.args.num_generations == plan["num_generations"]
        assert harness.trainer.args.per_device_train_batch_size == plan["micro_batch_size"]
        assert (
            harness.trainer.args.gradient_accumulation_steps == plan["gradient_accumulation_steps"]
        )
        assert harness.summary["num_generations"] == plan["num_generations"]

    def test_a_fresh_run_starts_from_no_checkpoint(self, tmp_path: Path, stubbed_run: object):
        del stubbed_run
        harness = run_arm(tmp_path)
        assert harness.trainer.resumed_from is None

    def test_the_estimator_defaults_reach_trl_and_the_launch_record(
        self, tmp_path: Path, stubbed_run: object
    ):
        """The shared defaults, not a re-hardcoded value: what TRL trains under is what is recorded.

        Before this wiring the two knobs existed only inside training_args.bin, so nothing in
        run_config.json said which estimator an arm's steps were trained under.
        """
        del stubbed_run
        harness = run_arm(tmp_path)
        assert harness.trainer.args.loss_type == estimator_defaults.GRPO_LOSS_TYPE
        assert harness.trainer.args.scale_rewards == estimator_defaults.GRPO_SCALE_REWARDS
        assert harness.run_config["config"]["loss_type"] == estimator_defaults.GRPO_LOSS_TYPE
        assert (
            harness.run_config["config"]["scale_rewards"] == estimator_defaults.GRPO_SCALE_REWARDS
        )

    def test_an_estimator_override_reaches_trl(self, tmp_path: Path, stubbed_run: object):
        """The A/B lever: one flag flips the estimator, and the launch record says which arm ran.

        "dapo" under Liger needs the acknowledgement since the faithfulness guard landed; the
        guard itself has its own test class below.
        """
        del stubbed_run
        harness = run_arm(
            tmp_path,
            loss_type="dapo",
            scale_rewards="batch",
            acknowledge_liger_estimator_mismatch=True,
        )
        assert harness.trainer.args.loss_type == "dapo"
        assert harness.trainer.args.scale_rewards == "batch"
        assert harness.run_config["config"]["scale_rewards"] == "batch"

    def test_an_unknown_estimator_value_is_refused_before_anything_loads(
        self, tmp_path: Path, stubbed_run: object
    ):
        """TRL would only raise at the first loss computation, on a loaded model on a rented GPU."""
        del stubbed_run
        with pytest.raises(ValueError, match="loss_type"):
            run_arm(tmp_path, loss_type="dpao")
        with pytest.raises(ValueError, match="scale_rewards"):
            run_arm(tmp_path, scale_rewards="bathc")

    def test_the_template_facts_it_measured_are_the_ones_it_recorded(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        harness = run_arm(tmp_path)
        assert harness.run_config["derived"]["prefilled_think"] is True
        assert harness.summary["prefilled_think"] is True

    def test_the_architecture_is_built_on_the_meta_device_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed_run: object
    ):
        """`count_meta_parameters` instantiates the whole model config-driven and is uncached.

        It was called twice, the second time only to format a log line that discarded the value it
        already had. At the 27B tier that is a second full module tree for a number in hand.
        """
        del stubbed_run
        builds: list[str] = []
        monkeypatch.setattr(
            gt,
            "count_meta_parameters",
            lambda model_id: builds.append(model_id) or 4_000_000_000,
        )
        harness = run_arm(tmp_path)
        assert builds == ["Qwen/Qwen3.5-4B"]
        assert harness.run_config["derived"]["meta_parameter_count"] == 4_000_000_000

    def test_the_tokenizers_ids_are_stamped_onto_what_save_model_will_write(
        self, tmp_path: Path, stubbed_run: object
    ):
        """Not what generation reads -- TRL overrides that -- but what a downstream eval load reads.

        The assignment used to sit behind `if hasattr(model, "generation_config")`, a branch that
        cannot be false: every transformers CausalLM carries one and PeftModel forwards the
        attribute. Dropping the guard means these four assignments are now unconditional, so the
        test that they happened is what stands in for it.
        """
        del stubbed_run
        harness = run_arm(tmp_path)
        model = harness.trainer.model
        assert model.config.eos_token_id == 151645
        assert model.generation_config.eos_token_id == 151645
        assert model.config.pad_token_id == 151645
        assert model.generation_config.pad_token_id == 151645

    def test_the_callbacks_include_the_shipper_only_when_a_destination_is_set(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        plain = run_arm(tmp_path, output_dir=str(tmp_path / "plain"))
        shipped = run_arm(
            tmp_path, output_dir=str(tmp_path / "shipped"), s3_dest="s3://bucket/prefix"
        )
        names = {type(c).__name__ for c in plain.trainer.build_kwargs["callbacks"]}
        shipped_names = {type(c).__name__ for c in shipped.trainer.build_kwargs["callbacks"]}
        assert "S3SyncCallback" not in names
        assert "S3SyncCallback" in shipped_names

    def test_retention_manifest_callback_is_opt_in_for_experiment_presets(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        plain = run_arm(tmp_path, output_dir=str(tmp_path / "plain"))
        retained = run_arm(
            tmp_path,
            output_dir=str(tmp_path / "retained"),
            record_retention_manifest=True,
            save_steps=1,
            save_total_limit=0,
        )
        plain_names = {
            type(callback).__name__ for callback in plain.trainer.build_kwargs["callbacks"]
        }
        retained_callbacks = retained.trainer.build_kwargs["callbacks"]
        retained_names = {type(callback).__name__ for callback in retained_callbacks}
        assert "CheckpointRetentionCallback" not in plain_names
        assert "CheckpointRetentionCallback" in retained_names
        retention_callback = next(
            callback
            for callback in retained_callbacks
            if isinstance(callback, gt.CheckpointRetentionCallback)
        )
        assert retention_callback.run_root == retained.output_dir
        assert retained.run_config["config"]["record_retention_manifest"] is True

    def test_every_built_trainer_carries_the_pace_guard(self, tmp_path: Path, stubbed_run: object):
        """The guard that kills a silently-slow run rides along unconditionally, s3 or not."""
        del stubbed_run
        harness = run_arm(tmp_path, output_dir=str(tmp_path / "paced"))
        names = [type(c).__name__ for c in harness.trainer.build_kwargs["callbacks"]]
        assert "StepPaceGuardCallback" in names

    def test_every_built_trainer_carries_the_phase_timer_and_attaches_it(
        self, tmp_path: Path, stubbed_run: object, monkeypatch: pytest.MonkeyPatch
    ):
        """The timer is both a callback (for the optimizer-step hooks) and attached to the built
        trainer (for the seams); a timer that was registered but never attached would write nothing."""
        del stubbed_run
        attached: list[object] = []
        monkeypatch.setattr(
            gt.StepPhaseTimer, "attach", lambda _self, trainer: attached.append(trainer)
        )
        harness = run_arm(tmp_path, output_dir=str(tmp_path / "timed"))
        callbacks = harness.trainer.build_kwargs["callbacks"]
        timers = [c for c in callbacks if isinstance(c, gt.StepPhaseTimer)]
        assert len(timers) == 1
        assert attached == [harness.trainer]

    def test_the_memory_monitor_lays_out_the_timing_and_trim_columns(
        self, tmp_path: Path, stubbed_run: object
    ):
        """One mem_log.csv row per step carries the phase seconds and the trim totals beside VRAM."""
        del stubbed_run
        harness = run_arm(tmp_path, output_dir=str(tmp_path / "columns"))
        monitors = [
            c
            for c in harness.trainer.build_kwargs["callbacks"]
            if type(c).__name__ == "MemoryMonitorCallback"
        ]
        assert len(monitors) == 1
        assert monitors[0].extra_columns == gt.MEM_LOG_EXTRA_COLUMNS
        assert "timing/generate_s" in monitors[0].extra_columns
        assert gt.STEP_TRIMMED_TOKENS_METRIC in monitors[0].extra_columns

    def test_the_attention_backends_are_appended_to_the_launch_record_after_the_build(
        self, tmp_path: Path, stubbed_run: object
    ):
        """`run_config.json` is written before the model loads and rewritten once the trainer
        exists, because which kernels a run trains and generates through is only knowable then.
        The rewrite must keep the launch's own facts: the same `started_at`, the same plan."""
        del stubbed_run
        harness = run_arm(tmp_path)
        record = harness.run_config
        assert record["derived"]["attention"] == STUB_ATTENTION
        assert record["derived"]["prefilled_think"] is True
        assert record["sizing_plan"]["num_generations"] == 4
        assert "started_at" in record
        assert not list(harness.output_dir.glob("*.tmp")), (
            "the atomic rewrite left its staging file"
        )

    def test_the_final_sync_runs_after_the_summary_exists(
        self, tmp_path: Path, stubbed_run: object, monkeypatch: pytest.MonkeyPatch
    ):
        """The callback's on_train_end fires inside trainer.train(), before save_model,
        save_state and the summary write, so the bucket's copy of every finished run was missing
        train_summary.json and the final state (observed on both 2026-08-19 arms). The ordering
        is the assertion: the recorder checks what is on disk at the moment the sync fires.
        """
        del stubbed_run
        synced: list[tuple[Path, str, bool]] = []

        def record_sync(local_dir: Path, s3_dest: str, **_kwargs: object) -> None:
            summary_on_disk = (local_dir / gt.TRAIN_SUMMARY_FILENAME).is_file()
            synced.append((local_dir, s3_dest, summary_on_disk))

        monkeypatch.setattr(gt, "sync_directory", record_sync)
        run = run_arm(tmp_path, output_dir=str(tmp_path / "shipped"), s3_dest="s3://bucket/prefix")
        assert synced, "the final sync never ran"
        local_dir, destination, summary_on_disk = synced[-1]
        assert local_dir == run.output_dir
        assert destination == "s3://bucket/prefix"
        assert summary_on_disk, "the final sync fired before train_summary.json was written"

    def test_a_run_with_no_destination_ships_nothing_at_the_end_either(
        self, tmp_path: Path, stubbed_run: object, monkeypatch: pytest.MonkeyPatch
    ):
        del stubbed_run
        synced: list[object] = []
        monkeypatch.setattr(gt, "sync_directory", lambda *a, **_k: synced.append(a))
        run_arm(tmp_path, output_dir=str(tmp_path / "plain"))
        assert synced == []


class TestTheLigerFaithfulnessGuardAndTheExecutedEstimatorRecord:
    """The two closures of the Liger-dapo audit: refuse the silent mismatch, record what executed.

    Every arm before 2026-08-20 recorded loss_type="dapo" while executing per-sequence GRPO,
    because TRL 1.10's Liger path drops num_items_in_batch. Nothing in the config surface or the
    artifacts said so. The guard makes that combination refuse at config construction -- before a
    tokenizer, corpus or rented GPU is touched -- and the acknowledgement that unblocks it is a
    recorded field, so a deliberate mismatch run announces itself in run_config.json forever.
    """

    def test_dapo_under_liger_refuses_at_config_construction(self, tmp_path: Path):
        """No stubs installed on purpose: the raise must precede every load this fixture fakes."""
        with pytest.raises(ValueError, match="num_items_in_batch"):
            gt.GameTrainConfig(
                arm="twin-pd-group",
                generate_fresh=True,
                output_dir=str(tmp_path / "run"),
                loss_type="dapo",
                scale_rewards="batch",
            )

    def test_the_refusal_covers_every_unfaithful_loss_type(self, tmp_path: Path):
        for loss_type in estimator_defaults.LIGER_UNFAITHFUL_LOSS_TYPES:
            with pytest.raises(ValueError, match="use_liger_kernel"):
                gt.GameTrainConfig(
                    arm="twin-pd-group",
                    generate_fresh=True,
                    output_dir=str(tmp_path / "run"),
                    loss_type=loss_type,
                )

    def test_the_acknowledgement_unblocks_and_lands_in_the_launch_record(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        harness = run_arm(
            tmp_path,
            loss_type="dapo",
            scale_rewards="batch",
            acknowledge_liger_estimator_mismatch=True,
        )
        assert harness.trainer.args.loss_type == "dapo"
        assert harness.run_config["config"]["acknowledge_liger_estimator_mismatch"] is True

    def test_no_liger_dapo_needs_no_acknowledgement(self, tmp_path: Path, stubbed_run: object):
        del stubbed_run
        harness = run_arm(tmp_path, loss_type="dapo", scale_rewards="batch", use_liger_kernel=False)
        assert harness.run_config["executed_estimator"] == "dapo (non-Liger TRL path)"

    def test_an_acknowledgement_with_nothing_to_acknowledge_is_refused(self, tmp_path: Path):
        """The default loss_type is faithful, so the flag on a copied command line is stale."""
        with pytest.raises(ValueError, match="no mismatch"):
            gt.GameTrainConfig(
                arm="twin-pd-group",
                generate_fresh=True,
                output_dir=str(tmp_path / "run"),
                acknowledge_liger_estimator_mismatch=True,
            )

    def test_the_default_run_records_a_faithful_executed_estimator(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        harness = run_arm(tmp_path)
        assert harness.run_config["executed_estimator"] == "dr_grpo (faithful under Liger)"

    def test_the_recorded_estimator_is_derived_from_the_recorded_shape(
        self, tmp_path: Path, stubbed_run: object
    ):
        """One source of truth: the record's string must be the pure function of the record's own
        loss_type, liger flag and micro-batch size -- never a second hand-written copy."""
        del stubbed_run
        harness = run_arm(
            tmp_path,
            loss_type="dapo",
            scale_rewards="batch",
            acknowledge_liger_estimator_mismatch=True,
        )
        record = harness.run_config
        assert record["executed_estimator"] == estimator_defaults.executed_estimator(
            record["config"]["loss_type"],
            use_liger_kernel=record["config"]["use_liger_kernel"],
            per_device_train_batch_size=record["sizing_plan"]["micro_batch_size"],
        )
        assert "per-sequence GRPO" in record["executed_estimator"] or (
            "per-micro-batch token normalization" in record["executed_estimator"]
        )

    def test_the_cli_flag_reaches_the_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(gt, "bridge_decode_kernel", lambda: {"bridged": False})
        monkeypatch.setattr(
            gt,
            "assert_bridged_kernel_matches_call_site",
            lambda: SimpleNamespace(positional_count=3, keyword_names=frozenset()),
        )
        monkeypatch.setattr(
            gt,
            "train_game_arm",
            lambda config, **_kwargs: seen.update(
                {"acknowledged": config.acknowledge_liger_estimator_mismatch}
            ),
        )
        gt.main(
            [
                "--arm",
                "dictator",
                "--generate-fresh",
                "--output-dir",
                str(tmp_path / "run"),
                "--loss-type",
                "dapo",
                "--scale-rewards",
                "batch",
                "--acknowledge-liger-estimator-mismatch",
            ]
        )
        assert seen["acknowledged"] is True


class TestMainBridgesTheDecodeKernelAndChecksIt:
    """`main` is where the DeltaNet decode kernel is bridged, and now where the bridge is verified.

    transformers filters the keyword arguments it passes down to the implementation's own parameter
    names and drops the rest silently, so a bridged kernel whose signature is missing one -- fla's
    `initial_state`, say -- runs fast and computes a different recurrence for every rollout of the
    run. The guard that catches it existed and was reachable only from a CLI nobody runs before a
    training job.
    """

    def stub_bridge(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Replace the bridge and its guard with recorders, and swallow the training call."""
        calls: list[str] = []

        def bridge() -> dict[str, object]:
            calls.append("bridge")
            return {"bridged": True, "implementation": "fla (stub)"}

        def guard() -> SimpleNamespace:
            calls.append("guard")
            return SimpleNamespace(positional_count=3, keyword_names=frozenset({"initial_state"}))

        monkeypatch.setattr(gt, "bridge_decode_kernel", bridge)
        monkeypatch.setattr(gt, "assert_bridged_kernel_matches_call_site", guard)
        return calls

    def test_the_call_site_is_checked_after_the_bridge_and_before_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls = self.stub_bridge(monkeypatch)
        monkeypatch.setattr(
            gt, "train_game_arm", lambda _config, **_kwargs: calls.append("train") or None
        )
        gt.main(["--arm", "dictator", "--generate-fresh", "--output-dir", str(tmp_path / "run")])
        assert calls == ["bridge", "guard", "train"]

    def test_the_run_record_carries_what_the_bridge_and_the_guard_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        self.stub_bridge(monkeypatch)
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            gt,
            "train_game_arm",
            lambda _config, *, kernel_bridge: seen.update(cast("dict[str, object]", kernel_bridge)),
        )
        gt.main(["--arm", "dictator", "--generate-fresh", "--output-dir", str(tmp_path / "run")])
        assert seen["bridged"] is True
        assert seen["decode_call_site"] == {
            "positional_count": 3,
            "keyword_names": ["initial_state"],
        }


class TestTheScalarEosRepairReachesTheTrainer:
    """TRL builds its GenerationConfig from the tokenizer at construction, so the repair is early.

    `openbmb/MiniCPM5-1B` declares stop tokens [1, 130073] while its scalar `eos_token` is id 1,
    which its template never emits. Unrepaired, nothing halts: every rollout runs to the completion
    cap, reads as truncated thinking, and earns the parse penalty until the reward's whole-batch
    raise fires -- after a model load and a full generation batch have been paid for.
    """

    def test_the_repair_runs_before_the_trainer_is_constructed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed_run: object
    ):
        del stubbed_run
        order: list[str] = []

        def repair(_tokenizer: object, model_id: str) -> dict[str, object]:
            order.append("repair")
            return {"repaired": True, "reason": f"stub for {model_id}"}

        class OrderedTrainer(StubTrainer):
            def __init__(self, **kwargs: Any) -> None:
                order.append("trainer")
                super().__init__(**kwargs)

        monkeypatch.setattr(preflight, "repair_scalar_eos", repair)
        monkeypatch.setattr(gt, "DynamicSampledGRPOTrainer", OrderedTrainer)
        run_arm(tmp_path)
        assert order == ["repair", "trainer"]

    def test_what_the_repair_reported_lands_in_the_run_record(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        harness = run_arm(tmp_path)
        assert harness.run_config["derived"]["scalar_eos"]["reason"].startswith("stub for")


class TestAResumeIsCheckedAgainstTheLaunchItContinues:
    """The critical one: a resumed arm must not silently become a different experiment.

    Every live sequence runner passes `--resume-from-checkpoint latest` with an explicit
    `--output-dir`, and the free-VRAM reading the sizing plan is derived from is a live
    `mem_get_info` call. Two gibibytes held by another process on the same card is enough to
    re-plan `num_generations` from 8 to 7 -- which is both the GRPO advantage baseline and the
    population group-mix grading estimates the opponent distribution from -- while the only record
    of the original value is overwritten in the same breath.
    """

    def seed_a_checkpoint(
        self, harness: RunHarness, step: int = 1, *, omit: tuple[str, ...] = ()
    ) -> Path:
        """Lay down a complete checkpoint next to a completed run's artifacts, as a save would.

        Every file the completeness gate requires is present unless named in `omit`, which is how a
        test builds the torn checkpoint a reclaim mid-sync leaves behind. The default step sits
        below the harness's `max_steps` of 2, so that resuming it has work left to do; a checkpoint
        AT `max_steps` is the already-complete relaunch, which has its own class below.
        """
        return seed_checkpoint(harness.output_dir, step, omit=omit)

    def test_a_matching_resume_continues_from_the_checkpoint(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        first = run_arm(tmp_path)
        checkpoint = self.seed_a_checkpoint(first)
        second = run_arm(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)
        assert second.trainer.resumed_from == str(checkpoint)
        resumed_record = json.loads(
            next(second.output_dir.glob("run_config.resume-from-*.json")).read_text()
        )
        assert resumed_record["derived"]["incomplete_checkpoints_set_aside"] == []

    def test_a_torn_newest_checkpoint_falls_back_to_the_complete_one_and_says_so(
        self, tmp_path: Path, stubbed_run: object, caplog: pytest.LogCaptureFixture
    ):
        """A reclaim mid-sync leaves `checkpoint-2` without its optimizer. Resuming it would
        restart Adam at step 2 with every signal green; the run resumes `checkpoint-1` instead,
        names the fallback in its log and in the resumed launch record, and moves the torn
        directory out of the `checkpoint-*` glob the eval batteries read."""
        del stubbed_run
        first = run_arm(tmp_path)
        complete = self.seed_a_checkpoint(first, step=1)
        torn = self.seed_a_checkpoint(first, step=2, omit=("optimizer.pt",))
        with caplog.at_level("WARNING", logger="games.train"):
            second = run_arm(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)
        assert second.trainer.resumed_from == str(complete)
        assert "INCOMPLETE CHECKPOINT: checkpoint-2 lacks ['optimizer.pt']" in caplog.text
        assert not torn.exists()
        resumed_record = json.loads(
            next(second.output_dir.glob("run_config.resume-from-checkpoint-1.json")).read_text()
        )
        (aside,) = resumed_record["derived"]["incomplete_checkpoints_set_aside"]
        assert aside["checkpoint"] == "checkpoint-2"
        assert aside["step"] == 2
        assert aside["missing"] == ["optimizer.pt"]
        assert Path(aside["moved_to"]).is_dir()
        assert Path(aside["moved_to"]).parent.parent.name == gt.INCOMPLETE_CHECKPOINTS_DIRNAME

    def test_a_bitten_but_sufficient_card_resumes_the_identical_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed_run: object
    ):
        """The same command, on a card another process has taken a bite out of.

        Under colocate the episodes decode inside the engine's reservation, so the group size no
        longer re-plans from a live free-VRAM reading -- the bite must leave the plan exactly as
        recorded, or the resume check would refuse a legitimate restart after any small leak.
        """
        del stubbed_run
        first = run_arm(tmp_path)
        checkpoint = self.seed_a_checkpoint(first)
        recorded_plan = first.run_config["sizing_plan"]
        assert recorded_plan["num_generations"] == 4
        crowded = dict(STUB_DEVICE)
        crowded["free_vram_gib_at_start"] = free_vram_leaving_room_for(recorded_plan, episodes=2.5)
        monkeypatch.setattr(gt, "describe_device", lambda: dict(crowded))
        second = run_arm(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)
        assert second.trainer.resumed_from == str(checkpoint)

    def test_a_card_that_cannot_hold_the_trainer_refuses_before_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed_run: object
    ):
        """The sabotage that is still reachable: a bite so large the weights no longer fit.

        The refusal comes from the sizing plan itself, before the resume continues anything, and
        the recorded launch stays untouched for the retry on a freer card.
        """
        del stubbed_run
        first = run_arm(tmp_path)
        self.seed_a_checkpoint(first)
        recorded_plan = first.run_config["sizing_plan"]
        fixed_gib = recorded_plan["usable_vram_gib"] - recorded_plan["episode_headroom_gib"]
        crowded = dict(STUB_DEVICE)
        crowded["free_vram_gib_at_start"] = (
            0.5 * fixed_gib / sizing.DEFAULT_VRAM_USABLE_FRACTION
            + recorded_plan["engine_reserved_gib"]
        )
        monkeypatch.setattr(gt, "describe_device", lambda: dict(crowded))
        with pytest.raises(RuntimeError, match="nothing left for episodes"):
            run_arm(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)
        assert first.run_config["sizing_plan"]["num_generations"] == 4

    def test_a_resume_under_a_different_arm_is_refused(self, tmp_path: Path, stubbed_run: object):
        del stubbed_run
        first = run_arm(tmp_path)
        self.seed_a_checkpoint(first)
        with pytest.raises(RuntimeError, match="arm"):
            run_arm(tmp_path, arm="twin-pd-self", resume_from_checkpoint=gt.RESUME_LATEST)

    def test_a_resume_under_a_different_model_is_refused(self, tmp_path: Path, stubbed_run: object):
        del stubbed_run
        first = run_arm(tmp_path)
        self.seed_a_checkpoint(first)
        # An unscreened checkpoint, so the completion-budget floor stays the generic one and this
        # launch differs from the recorded one in the model alone.
        with pytest.raises(RuntimeError, match="model_id"):
            run_arm(
                tmp_path,
                model_id="Qwen/Qwen3.6-27B",
                resume_from_checkpoint=gt.RESUME_LATEST,
            )

    def test_the_identity_refusal_costs_no_model_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed_run: object
    ):
        """Refused before the tokenizer is read, so the check is free rather than merely cheap."""
        del stubbed_run
        first = run_arm(tmp_path)
        self.seed_a_checkpoint(first)

        def refuse_to_load(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("the resume check must refuse before the tokenizer is loaded")

        monkeypatch.setattr(
            preflight.AutoTokenizer,
            "from_pretrained",
            classmethod(lambda _cls, *a, **k: refuse_to_load()),
        )
        with pytest.raises(RuntimeError, match="arm"):
            run_arm(tmp_path, arm="twin-pd-self", resume_from_checkpoint=gt.RESUME_LATEST)

    def test_the_original_launch_record_survives_a_resume(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        first = run_arm(tmp_path)
        first_started = first.run_config["started_at"]
        self.seed_a_checkpoint(first)
        second = run_arm(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)
        assert second.run_config["started_at"] == first_started
        resumed = list(second.output_dir.glob("run_config.resume-from-*.json"))
        assert len(resumed) == 1
        assert json.loads(resumed[0].read_text())["started_at"] != first_started

    def strip_estimator_fields(self, harness: RunHarness) -> None:
        """Rewrite a run's launch record as the pre-estimator-field era would have written it."""
        record_path = harness.output_dir / gt.RUN_CONFIG_FILENAME
        record = json.loads(record_path.read_text())
        for field in gt.RESUME_IDENTITY_DEFAULTS:
            del record["config"][field]
        record_path.write_text(json.dumps(record))

    def test_a_pre_estimator_record_resumes_under_the_values_its_era_hardcoded(
        self, tmp_path: Path, stubbed_run: object
    ):
        """Absence in an old record means dapo/batch -- the values the hardcoded lines held --
        so a launch stating those continues, rather than tripping on absence != anything."""
        del stubbed_run
        first = run_arm(
            tmp_path,
            loss_type="dapo",
            scale_rewards="batch",
            acknowledge_liger_estimator_mismatch=True,
        )
        checkpoint = self.seed_a_checkpoint(first)
        self.strip_estimator_fields(first)
        second = run_arm(
            tmp_path,
            loss_type="dapo",
            scale_rewards="batch",
            acknowledge_liger_estimator_mismatch=True,
            resume_from_checkpoint=gt.RESUME_LATEST,
        )
        assert second.trainer.resumed_from == str(checkpoint)

    def test_a_pre_estimator_record_refuses_a_resume_under_the_new_defaults(
        self, tmp_path: Path, stubbed_run: object
    ):
        """The checkpoint's steps were trained under dapo/batch; continuing them under the current
        defaults would put two estimators' steps under one run's step numbers, silently."""
        del stubbed_run
        first = run_arm(
            tmp_path,
            loss_type="dapo",
            scale_rewards="batch",
            acknowledge_liger_estimator_mismatch=True,
        )
        self.seed_a_checkpoint(first)
        self.strip_estimator_fields(first)
        with pytest.raises(RuntimeError, match="scale_rewards"):
            run_arm(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)

    def test_resuming_a_checkpoint_with_no_launch_record_is_refused(
        self, tmp_path: Path, stubbed_run: object
    ):
        """A checkpoint whose run_config.json is gone cannot be checked, so it is not resumed."""
        del stubbed_run
        first = run_arm(tmp_path)
        self.seed_a_checkpoint(first)
        (first.output_dir / gt.RUN_CONFIG_FILENAME).unlink()
        with pytest.raises(RuntimeError, match=gt.RUN_CONFIG_FILENAME):
            run_arm(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)

    def test_latest_with_nothing_to_resume_writes_a_fresh_launch_record(
        self, tmp_path: Path, stubbed_run: object
    ):
        """`latest` on an empty directory is a first launch, which is what makes re-running safe."""
        del stubbed_run
        harness = run_arm(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)
        assert harness.trainer.resumed_from is None
        assert not list(harness.output_dir.glob("run_config.resume-from-*.json"))


class TestAFrozenOpponentArmNeedsItsOpponentBeforeAModelLoads:
    """`--generate-fresh` writes opp_coop_prob=-1, which the reward rejects one batch too late."""

    def test_a_fresh_generation_for_a_vs_fixed_mix_arm_is_refused_at_startup(self):
        with pytest.raises(ValueError, match="opp_coop_prob"):
            gt.prepare_rows(
                gt.GameTrainConfig(arm="pd-vs-frozen", generate_fresh=True, max_prompts=4)
            )

    def test_the_refusal_names_the_sweep_stage_that_fills_it(self):
        with pytest.raises(ValueError, match="frozen-opponent-model"):
            gt.prepare_rows(gt.GameTrainConfig(arm="stag-hunt-vs-frozen", generate_fresh=True))

    def test_a_corpus_whose_probabilities_were_never_sampled_is_refused(self, tmp_path: Path):
        rows = [
            {
                "prompt": "a sheet of paper",
                "prompt_id": f"pd-vs-frozen-{index}",
                "game_id": "pd-vs-frozen",
                "grading": "vs-fixed-mix",
                "payoff_variant": "temptation-2",
                "opp_coop_prob": -1.0,
            }
            for index in range(3)
        ]
        path = tmp_path / "corpus.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        config = gt.GameTrainConfig(arm="pd-vs-frozen", corpus_path=str(path))
        with pytest.raises(ValueError, match="outside"):
            gt.prepare_rows(config)

    def test_a_filled_corpus_passes(self, tmp_path: Path):
        rows = [
            {
                "prompt": "a sheet of paper",
                "prompt_id": f"pd-vs-frozen-{index}",
                "game_id": "pd-vs-frozen",
                "grading": "vs-fixed-mix",
                "payoff_variant": "temptation-2",
                "opp_coop_prob": 0.42,
            }
            for index in range(3)
        ]
        path = tmp_path / "corpus.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        config = gt.GameTrainConfig(arm="pd-vs-frozen", corpus_path=str(path))
        assert len(gt.prepare_rows(config)) == 3

    def test_an_arm_with_no_frozen_opponent_is_left_alone(self):
        rows = gt.prepare_rows(
            gt.GameTrainConfig(arm="twin-pd-group", generate_fresh=True, max_prompts=4)
        )
        assert len(rows) == 4


class TestTheDatasetHoldsAWholeStepOfPrompts:
    """TRL's RepeatSampler drops every incomplete chunk, so a short dataset yields no steps at all.

    The sampler chunks the shuffled index list into `generation_batch_size // num_generations`
    prompts and discards any remainder, and that quotient is exactly `plan.prompts_per_step`. Below
    it the run performs no optimizer step and surfaces as "no training record carried a reward",
    minutes and a model load later.
    """

    def test_fewer_prompts_than_one_step_needs_is_refused(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        with pytest.raises(RuntimeError, match="prompts_per_step"):
            run_arm(tmp_path, max_prompts=1, prompts_per_step=4, num_generations=4)

    def test_exactly_one_step_of_prompts_is_enough(self, tmp_path: Path, stubbed_run: object):
        del stubbed_run
        harness = run_arm(tmp_path, max_prompts=4, prompts_per_step=4, num_generations=4)
        assert harness.run_config["derived"]["n_prompts"] == 4


class TestARelaunchOfAFinishedRunTouchesNothing:
    """The 2026-09-01 incident, offline: `--resume-from-checkpoint latest` onto a run at max_steps.

    Left to the trainer, transformers restores `global_step == max_steps`, trains zero steps, and
    still logs once at train end -- which wrote a zero-row parquet over step 70's 64 completions,
    opened a fresh `mem_log.csv`, and re-reported the run with a 40-second wall clock. The relaunch
    is now recognised before the trainer exists, and the assertion is byte-level: the run
    directory after the relaunch is identical to the run directory before it.
    """

    def test_a_resume_landing_on_max_steps_builds_no_trainer_and_changes_no_file(
        self,
        tmp_path: Path,
        stubbed_run: object,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        del stubbed_run
        first = run_arm(tmp_path)
        seed_checkpoint(first.output_dir, 2)
        before = snapshot_tree(first.output_dir)
        trainers_built = len(StubTrainer.instances)
        synced: list[object] = []
        monkeypatch.setattr(gt, "sync_directory", lambda *a, **_k: synced.append(a))
        with caplog.at_level("INFO", logger="games.train"):
            result = gt.train_game_arm(
                arm_config(
                    tmp_path, resume_from_checkpoint=gt.RESUME_LATEST, s3_dest="s3://bucket/prefix"
                )
            )
        assert result is None
        assert len(StubTrainer.instances) == trainers_built, "a trainer was built for nothing"
        assert snapshot_tree(first.output_dir) == before
        assert not list(first.output_dir.glob("run_config.resume-from-*.json"))
        assert synced == []
        assert "ALREADY COMPLETE" in caplog.text
        assert "checkpoint-2 holds step 2 of max_steps=2" in caplog.text

    def test_a_checkpoint_past_max_steps_is_complete_too(self, tmp_path: Path, stubbed_run: object):
        """An operator who lowered --max-steps on relaunch has nothing to train either."""
        del stubbed_run
        first = run_arm(tmp_path)
        seed_checkpoint(first.output_dir, 3)
        before = snapshot_tree(first.output_dir)
        assert (
            gt.train_game_arm(arm_config(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)) is None
        )
        assert snapshot_tree(first.output_dir) == before

    def test_a_checkpoint_below_max_steps_still_resumes(self, tmp_path: Path, stubbed_run: object):
        """The gate keys on the step, so an unfinished run is untouched by it."""
        del stubbed_run
        first = run_arm(tmp_path)
        checkpoint = seed_checkpoint(first.output_dir, 1)
        second = run_arm(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST)
        assert second.trainer.resumed_from == str(checkpoint)

    def test_a_finished_run_whose_summary_never_landed_is_refused_not_declared_complete(
        self, tmp_path: Path, stubbed_run: object
    ):
        """Killed between the final save and the summary write: the checkpoints are the real output,
        and a zero-step relaunch would manufacture a wrong summary, so the launch refuses instead."""
        del stubbed_run
        first = run_arm(tmp_path)
        seed_checkpoint(first.output_dir, 2)
        (first.output_dir / gt.TRAIN_SUMMARY_FILENAME).unlink()
        before = snapshot_tree(first.output_dir)
        trainers_built = len(StubTrainer.instances)
        with pytest.raises(RuntimeError, match="does not exist") as excinfo:
            gt.train_game_arm(arm_config(tmp_path, resume_from_checkpoint=gt.RESUME_LATEST))
        assert gt.TRAIN_SUMMARY_FILENAME in str(excinfo.value)
        assert "train zero steps" in str(excinfo.value)
        assert len(StubTrainer.instances) == trainers_built
        assert snapshot_tree(first.output_dir) == before

    def test_the_identity_check_still_comes_first(self, tmp_path: Path, stubbed_run: object):
        """A relaunch under the wrong arm onto a finished run is refused as a mismatch, not waved
        through as already complete: the mismatch is the more useful thing to know."""
        del stubbed_run
        first = run_arm(tmp_path)
        seed_checkpoint(first.output_dir, 2)
        with pytest.raises(RuntimeError, match="refusing to resume"):
            gt.train_game_arm(
                arm_config(tmp_path, arm="twin-pd-self", resume_from_checkpoint=gt.RESUME_LATEST)
            )


class TestTheSummaryReadsTheTraceFilesItClaims:
    """`rollout_trace_complete` used to be config arithmetic; the incident's summary said true over
    a zero-row file. It now also reads the completions directory, and a gap fails the run."""

    def test_a_healthy_run_reports_the_trace_complete_from_its_files(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        harness = run_arm(tmp_path)
        assert harness.summary["rollout_trace_complete"] is True
        assert harness.summary["rollout_trace_config_complete"] is True
        files = harness.summary["rollout_trace_files"]
        assert files["steps_expected"] == 2
        assert files["expected_rows_per_step"] == harness.summary["episodes_per_step"]
        assert files["missing_steps"] == []
        assert files["empty_steps"] == []
        assert files["wrong_row_count_steps"] == []
        assert files["checked_dir"] == str(harness.output_dir / preflight.COMPLETIONS_DIRNAME)

    def test_an_empty_file_and_a_missing_step_turn_the_summary_red_and_fail_the_run(
        self, tmp_path: Path, stubbed_run: object, monkeypatch: pytest.MonkeyPatch
    ):
        """The sabotage: step 1 never reaches disk and step 2 lands as the incident's empty file."""
        del stubbed_run
        monkeypatch.setattr(StubTrainer, "trace_steps", (2,))
        monkeypatch.setattr(StubTrainer, "trace_rows", {2: 0})
        with pytest.raises(RuntimeError, match="rollout trace") as excinfo:
            run_arm(tmp_path)
        assert "missing steps [1]" in str(excinfo.value)
        assert "empty files at steps [2]" in str(excinfo.value)
        summary = json.loads((tmp_path / "run" / gt.TRAIN_SUMMARY_FILENAME).read_text())
        assert summary["rollout_trace_complete"] is False
        assert summary["rollout_trace_config_complete"] is True
        assert summary["rollout_trace_files"]["missing_steps"] == [1]
        assert summary["rollout_trace_files"]["empty_steps"] == [2]

    def test_a_short_file_is_reported_with_its_row_count(
        self, tmp_path: Path, stubbed_run: object, monkeypatch: pytest.MonkeyPatch
    ):
        del stubbed_run
        monkeypatch.setattr(StubTrainer, "trace_rows", {1: 3})
        with pytest.raises(RuntimeError, match=r"wrong row counts \(step, rows\) \[\[1, 3\]\]"):
            run_arm(tmp_path)
        summary = json.loads((tmp_path / "run" / gt.TRAIN_SUMMARY_FILENAME).read_text())
        assert summary["rollout_trace_files"]["wrong_row_count_steps"] == [[1, 3]]

    def test_the_red_summary_is_shipped_before_the_run_fails(
        self, tmp_path: Path, stubbed_run: object, monkeypatch: pytest.MonkeyPatch
    ):
        """Same ordering as missing metrics: the summary naming the gap is the artifact worth
        having off-box, so the final sync runs before the raise."""
        del stubbed_run
        monkeypatch.setattr(StubTrainer, "trace_steps", (2,))
        synced: list[bool] = []

        def record_sync(local_dir: Path, _s3_dest: str, **_kwargs: object) -> None:
            synced.append((local_dir / gt.TRAIN_SUMMARY_FILENAME).is_file())

        monkeypatch.setattr(gt, "sync_directory", record_sync)
        with pytest.raises(RuntimeError, match="rollout trace"):
            run_arm(tmp_path, output_dir=str(tmp_path / "shipped"), s3_dest="s3://bucket/prefix")
        assert synced == [True]


class TestTheParsePenaltyReachesTheReward:
    """`--parse-penalty` was recorded and resume-gated while the reward kept its own default.

    So a run could name a penalty in `run_config.json`, print it in the plan, refuse a resume that
    changed it, and still price every unparseable completion at the reward module's -1.0. The knob
    is the one lever the parse-versus-strategy ratio has, and wave 4b's kit sets it from a probe, so
    the value has to arrive at `make_game_reward` rather than only in the record.
    """

    @staticmethod
    def captured_penalty(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
    ) -> float:
        """Run one stubbed arm and report the parse penalty its reward was actually built with."""
        seen: list[float] = []
        built = gt.make_game_reward

        def record(num_generations: int, **kwargs: object) -> object:
            seen.append(cast("float", kwargs["parse_penalty"]))
            return built(num_generations, **kwargs)  # pyright: ignore[reportArgumentType]

        monkeypatch.setattr(gt, "make_game_reward", record)
        run_arm(tmp_path, **overrides)
        assert len(seen) == 1
        return seen[0]

    def test_the_configured_penalty_is_the_one_the_reward_prices_with(
        self, tmp_path: Path, stubbed_run: object, monkeypatch: pytest.MonkeyPatch
    ):
        del stubbed_run
        assert self.captured_penalty(tmp_path, monkeypatch, parse_penalty=-0.25) == pytest.approx(
            -0.25
        )

    def test_an_unset_penalty_arrives_as_the_value_every_banked_arm_trained_under(
        self, tmp_path: Path, stubbed_run: object, monkeypatch: pytest.MonkeyPatch
    ):
        del stubbed_run
        assert self.captured_penalty(tmp_path, monkeypatch) == pytest.approx(-1.0)

    def test_the_run_record_and_the_reward_cannot_disagree(
        self, tmp_path: Path, stubbed_run: object, monkeypatch: pytest.MonkeyPatch
    ):
        """The record is what a readout reads the penalty off, so it names the priced value."""
        del stubbed_run
        penalty = self.captured_penalty(tmp_path, monkeypatch, parse_penalty=-0.5)
        recorded = json.loads((tmp_path / "run" / gt.RUN_CONFIG_FILENAME).read_text())
        assert recorded["config"]["parse_penalty"] == pytest.approx(penalty)


class TestTheAdapterDropoutReachesPeft:
    """The dropout was recorded, resume-gated and plan-grepped while nothing read it at the adapter.

    Exactly the shape `--parse-penalty` failed in above, on the knob wave 4b's 9B arm sets to zero:
    `run_config.json` would name 0.0, the resume gate would refuse a relaunch that changed it and the
    kit's plan grep would pass, while `LoraConfig` took peft's own default and the graded forward
    dropped units the rollout never dropped. So the assertion is at the consumption point, the
    `peft_config` the trainer was built with.
    """

    @staticmethod
    def built_dropout(tmp_path: Path, **overrides: object) -> float:
        """Run one stubbed arm and report the dropout the LoRA adapter was actually built with."""
        harness = run_arm(tmp_path, **overrides)
        return cast("float", harness.trainer.build_kwargs["peft_config"].lora_dropout)

    def test_a_configured_zero_reaches_the_adapter(self, tmp_path: Path, stubbed_run: object):
        """Zero rather than a truthy value on purpose: `or`-style plumbing would substitute 0.05."""
        del stubbed_run
        assert self.built_dropout(tmp_path, lora_dropout=0.0) == pytest.approx(0.0)

    def test_an_unset_dropout_arrives_as_the_value_every_banked_arm_trained_under(
        self, tmp_path: Path, stubbed_run: object
    ):
        del stubbed_run
        assert self.built_dropout(tmp_path) == pytest.approx(0.05)

    def test_the_run_record_and_the_adapter_cannot_disagree(
        self, tmp_path: Path, stubbed_run: object
    ):
        """The record is what the resume gate and every readout read the dropout off."""
        del stubbed_run
        harness = run_arm(tmp_path, lora_dropout=0.0)
        built = cast("float", harness.trainer.build_kwargs["peft_config"].lora_dropout)
        assert harness.run_config["config"]["lora_dropout"] == pytest.approx(built)

    def test_the_rank_and_alpha_reach_it_too(self, tmp_path: Path, stubbed_run: object):
        """The other two adapter knobs travel the same line, and a resume across either re-scales it."""
        del stubbed_run
        harness = run_arm(tmp_path, lora_rank=8, lora_alpha=64)
        peft_config = harness.trainer.build_kwargs["peft_config"]
        assert (peft_config.r, peft_config.lora_alpha) == (8, 64)
