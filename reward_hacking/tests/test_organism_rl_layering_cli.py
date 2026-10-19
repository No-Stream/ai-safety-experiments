"""CLI contracts for layered organism training runs."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file

from reward_hacking.organism_rl import cli
from reward_hacking.tests.test_organism_rl_adapter_init import _checkpoint, _config, _model


def test_layering_defaults() -> None:
    args = cli.parse_args(["--output-dir", "artifacts/toy"])
    assert args.init_adapter_from is None
    assert args.lr_scheduler_type == "constant"
    assert args.warmup_steps == 0
    assert args.prompt_pool_max_pass_fraction == 1.0


@pytest.mark.parametrize("scheduler", ["linear", "constant", "constant_with_warmup", "cosine"])
def test_scheduler_flags_reach_grpo_config(scheduler: str) -> None:
    args = cli.parse_args(
        ["--output-dir", "artifacts/toy", "--lr-scheduler-type", scheduler, "--warmup-steps", "3"]
    )
    config = cli.build_grpo_config(args, 128, 2, device=torch.device("cpu"))
    assert config.lr_scheduler_type == scheduler
    assert config.warmup_steps == 3


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--lr-scheduler-type", "linear"),
        ("--warmup-steps", "3"),
        ("--prompt-pool-max-pass-fraction", "0.99"),
    ],
)
def test_new_semantic_flags_refuse_resume_drift(tmp_path: Path, flag: str, value: str) -> None:
    argv = ["--output-dir", str(tmp_path)]
    cli.write_provenance(cli.parse_args(argv), {"name": "cpu"})
    with pytest.raises(ValueError, match=flag.removeprefix("--").replace("-", "_")):
        cli.write_provenance(cli.parse_args([*argv, flag, value]), {"name": "cpu"})


def test_legacy_flags_normalize_to_layering_defaults() -> None:
    flags = vars(cli.parse_args(["--output-dir", "artifacts/toy"]))
    legacy = {
        key: value
        for key, value in flags.items()
        if key
        not in {
            "init_adapter_from",
            "lr_scheduler_type",
            "warmup_steps",
            "prompt_pool_max_pass_fraction",
        }
    }
    # Runs recorded before the flag existed used TRL's linear decay; resuming them under the constant default is drift.
    legacy_semantics = cli._semantic_flags(legacy)
    current_semantics = cli._semantic_flags(flags)
    assert legacy_semantics.pop("lr_scheduler_type") == "linear"
    assert current_semantics.pop("lr_scheduler_type") == "constant"
    assert legacy_semantics == current_semantics


def test_init_adapter_and_exact_resume_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--init-adapter-from",
                "checkpoint-1",
                "--resume-from-checkpoint",
                "checkpoint-2",
            ]
        )
    assert "cannot be combined" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--warmup-steps", "-1"),
        ("--prompt-pool-max-pass-fraction", "-0.1"),
        ("--prompt-pool-max-pass-fraction", "1.1"),
        ("--prompt-pool-max-pass-fraction", "nan"),
    ],
)
def test_invalid_layering_flags_refused(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--output-dir", "artifacts/toy", flag, value])


def test_exact_resume_retains_layering_lineage_without_reloading_source(tmp_path: Path) -> None:
    args = cli.parse_args(["--output-dir", str(tmp_path)])
    path = cli.write_provenance(args, {"name": "cpu"})
    stored = json.loads(path.read_text())
    stored["flags"]["init_adapter_from"] = "removed-source/checkpoint-3"
    stored["init_adapter_from"] = {"lineage": [{"source_step": 3}]}
    path.write_text(json.dumps(stored))
    args.resume_from_checkpoint = tmp_path / "checkpoint-1"
    cli.write_provenance(args, {"name": "cpu"})
    assert json.loads(path.read_text())["init_adapter_from"] == stored["init_adapter_from"]


def test_new_run_initialization_reaches_training_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path / "source")
    args = cli.parse_args(
        [
            "--output-dir",
            str(tmp_path / "new-run"),
            "--model",
            "synthetic-base",
            "--endpoint",
            "http://server",
            "--init-adapter-from",
            str(checkpoint),
            "--mamba-ssm-cache-dtype",
            "float32",
        ]
    )
    model = _model()
    cast("Any", model.config).dtype = torch.float32
    cli.resolve_sampler_precision(args, model.config)
    monkeypatch.setattr(cli, "_training_lora_config", lambda _name: _config())
    path = cli.write_provenance(args, {"name": "cpu"}, model_revision="synthetic-revision")
    metadata = json.loads(path.read_text())["init_adapter_from"]
    assert metadata["source_checkpoint"] == str(checkpoint.resolve())
    assert metadata["lineage"][-1]["source_step"] == 7
    monkeypatch.setattr(cli, "load_base_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "get_peft_model", lambda *a, **k: model)
    monkeypatch.setattr(cli, "install_before_load", lambda *_a: None)
    monkeypatch.setattr(
        cli,
        "memory_options_from_args",
        lambda _args: SimpleNamespace(
            prewarm_fla=False, base_quantization="none", tiled_mlp=False, offload_checkpoints=False
        ),
    )
    monkeypatch.setattr(
        cli,
        "VLLMPhaseController",
        lambda **_k: SimpleNamespace(sleep=lambda: None, close=lambda: None),
    )
    monkeypatch.setattr(cli, "OrganismRollout", lambda *a, **k: SimpleNamespace(close=lambda: None))
    config_builder = cli.build_grpo_config
    monkeypatch.setattr(
        cli,
        "build_grpo_config",
        lambda args, tokens, generations, **_k: config_builder(
            args, tokens, generations, device=torch.device("cpu")
        ),
    )
    captured: dict[str, Any] = {}

    class TrainingBoundary:
        def __init__(self, **kwargs: Any) -> None:
            self.model = kwargs["model"]
            self.optimizer = torch.optim.AdamW(
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            )
            self.state = SimpleNamespace(global_step=0)
            captured["config"] = kwargs["args"]

        def train(self, *, resume_from_checkpoint: str | None) -> None:
            assert resume_from_checkpoint is None
            assert self.state.global_step == 0
            assert self.optimizer.state == {}
            actual = get_peft_model_state_dict(self.model)
            for name, tensor in load_file(checkpoint / "adapter_model.safetensors").items():
                assert torch.equal(actual[name], tensor)
                assert actual[name].dtype == torch.float32
            captured["trained"] = True

    monkeypatch.setattr(cli, "OrganismGRPOTrainer", TrainingBoundary)
    source = SimpleNamespace(
        backend=SimpleNamespace(
            model_id="synthetic-base", tokenizer=object(), request_admission=None
        )
    )
    cli._run_training(
        args,
        endpoint="http://server",
        device_index=0,
        max_tokens=128,
        generations=2,
        rollout_concurrency=1,
        model_revision="synthetic-revision",
        source=cast("Any", source),
        task_ids=("synthetic",),
    )
    assert captured["trained"]
    assert captured["config"].output_dir == str(args.output_dir)


def test_pool_fraction_reaches_composition_log_and_provenance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    screen = tmp_path / "screen"
    screen.mkdir()
    records = [
        {
            "task_id": task,
            "replica": replica,
            "reward": reward,
            "infrastructure_failure": None,
            "family": "ilcb",
            "variant": variant,
        }
        for task, variant, rewards in [("easy", "original", [1, 1]), ("mixed", "oneoff", [1, 0])]
        for replica, reward in enumerate(rewards)
    ]
    (screen / "samples.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    args = cli.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--prompt-pool-from",
            str(screen),
            "--prompt-pool-max-pass-fraction",
            "0.99",
        ]
    )
    with caplog.at_level("INFO", logger=cli.__name__):
        selected, composition = cli._apply_prompt_pool(
            args,
            [cli.TaskPrompt("easy", "ilcb", "original"), cli.TaskPrompt("mixed", "ilcb", "oneoff")],
        )
    assert [prompt.task_id for prompt in selected] == ["mixed"]
    assert composition is not None
    assert composition["dropped"] == 1
    assert "'dropped': 1" in caplog.text
    path = cli.write_provenance(args, {"name": "cpu"}, prompt_pool=composition)
    assert (
        json.loads(path.read_text())["prompt_pool"]["by_family_variant"]["ilcb/original"]["dropped"]
        == 1
    )


def test_init_adapter_refuses_existing_run_output(tmp_path: Path) -> None:
    args = cli.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--init-adapter-from",
            str(tmp_path / "source/checkpoint-1"),
        ]
    )
    (args.output_dir / "checkpoint-2").mkdir(parents=True)
    with pytest.raises(ValueError, match="new output directory"):
        cli.write_provenance(args, {"name": "cpu"})


@pytest.mark.parametrize("output_suffix", ["", "checkpoint-1", "checkpoint-1/nested"])
def test_init_adapter_refuses_writing_into_source_directory(
    tmp_path: Path, output_suffix: str, capsys: pytest.CaptureFixture[str]
) -> None:
    source_run = tmp_path / "source"
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--output-dir",
                str(source_run / output_suffix),
                "--init-adapter-from",
                str(source_run / "checkpoint-1"),
            ]
        )
    assert "own new output directory" in capsys.readouterr().err
