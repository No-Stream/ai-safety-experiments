import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from reward_hacking.organism_rl import cli
from reward_hacking.organism_rl.cli import parse_args, resolve_sizing, write_provenance
from reward_hacking.organism_rl.precision import (
    assert_fp32_training_state,
    record_precision_provenance,
)
from reward_hacking.vllm_server import VLLMServerConfig


def test_defaults_and_vram_derived_sizing():
    args = parse_args(["--output-dir", "artifacts/toy", "--gpu-memory-utilization", "0.8"])
    assert args.temperature == args.top_p == 1.0
    assert args.families == ["ilcb", "aisi"]
    assert args.impossible_variant == "oneoff"
    assert args.solvable_fraction == 0.15
    assert args.in_band_fractions == {}
    assert args.ladder_fraction == 0
    assert args.checkpoint_keep_latest == 2
    assert args.checkpoint_keep_middle == 3
    assert args.fla_nb_bucket == "pow2"
    assert args.enable_prefix_caching
    assert args.rollout_concurrency is None
    assert args.expected_peak_context_tokens is None
    assert args.prompts_per_step == 1
    assert args.thinking_token_budget == 512
    assert args.max_consecutive_truncated_turns == 2
    assert args.kv_cache_dtype is None
    assert args.mamba_ssm_cache_dtype is None
    small = resolve_sizing(total_bytes=8 * 1024**3, hidden_size=1024, layers=12)
    large = resolve_sizing(total_bytes=16 * 1024**3, hidden_size=1024, layers=12)
    assert large.max_sequence_tokens > small.max_sequence_tokens
    assert small.micro_batch_size == 1


def test_checkpoint_retention_flags_are_operational(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--endpoint", "http://server:8000"])
    write_provenance(args, {"name": "cpu"})

    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--endpoint",
            "http://server:8000",
            "--checkpoint-keep-latest",
            "4",
            "--checkpoint-keep-middle",
            "0",
        ]
    )
    assert args.checkpoint_keep_latest == 4
    assert args.checkpoint_keep_middle == 0
    write_provenance(args, {"name": "cpu"})


@pytest.mark.parametrize(
    ("flag", "value"), [("--checkpoint-keep-latest", "0"), ("--checkpoint-keep-middle", "-1")]
)
def test_checkpoint_retention_rejects_invalid_counts(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "artifacts/toy", flag, value])


def test_task_mix_cli_accepts_repeatable_families_and_conflicting_ilcb() -> None:
    args = parse_args(
        [
            "--output-dir",
            "artifacts/toy",
            "--endpoint",
            "http://server:8000",
            "--family",
            "ilcb",
            "--family",
            "ilcb",
            "--impossible-variant",
            "conflicting",
            "--solvable-fraction",
            "0.25",
            "--seed",
            "41",
        ]
    )

    assert args.families == ["ilcb"]
    assert args.impossible_variant == "conflicting"
    assert args.solvable_fraction == 0.25
    assert args.seed == 41


@pytest.mark.parametrize("fraction", ["-0.1", "1", "nan", "inf"])
def test_task_mix_cli_rejects_invalid_solvable_fraction(fraction: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--endpoint",
                "http://server:8000",
                "--solvable-fraction",
                fraction,
            ]
        )


def test_task_mix_cli_rejects_conflicting_aisi() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--endpoint",
                "http://server:8000",
                "--family",
                "aisi",
                "--impossible-variant",
                "conflicting",
            ]
        )


def test_teacher_requires_stored_scores():
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "artifacts/toy", "--teacher-beta", "0.1"])


def test_owned_server_requires_an_explicit_gpu_memory_fraction() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "artifacts/toy"])


def test_provenance_has_flags_version_and_sha(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--gpu-memory-utilization", "0.8"])
    path = write_provenance(args, {"name": "cpu"}, model_revision="resolved-revision")
    value = json.loads(path.read_text())
    assert value["trl_version"] == "1.10.0"
    assert len(value["git_sha"]) == 40
    assert isinstance(value["git_dirty"], bool)
    assert isinstance(value["modified_tracked_paths"], list)
    assert value["model_revision"] == "resolved-revision"
    assert value["flags"]["temperature"] == 1.0
    assert value["flags"]["thinking_token_budget"] == 512
    assert value["flags"]["max_consecutive_truncated_turns"] == 2


def test_owned_server_options_build_the_requested_vllm_configuration(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps(
            {"before": {"mean_abs_logprob_gap": 0.01}, "after": {"mean_abs_logprob_gap": 0.02}}
        )
    )
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--model",
            "Qwen/test-model",
            "--model-revision",
            "resolved-revision",
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.75",
            "--max-lora-rank",
            "32",
            "--enable-prefix-caching",
            "--mamba-cache-mode",
            "align",
            "--kv-cache-dtype",
            "fp8",
            "--mamba-ssm-cache-dtype",
            "bfloat16",
            "--quantization-admission",
            str(admission),
            "--reasoning-config",
            '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}',
        ]
    )

    assert args.endpoint is None
    config = cli.build_server_config(args, model_revision="resolved-revision", max_model_len=4096)

    assert config.model == "Qwen/test-model"
    assert config.revision == "resolved-revision"
    assert config.max_model_len == 4096
    assert config.gpu_memory_utilization == 0.75
    assert config.enable_lora is True
    assert config.max_lora_rank == 32
    assert config.enable_sleep_mode is True
    assert config.language_model_only is True
    assert "--language-model-only" in config.command()
    assert json.loads(config.command()[config.command().index("--hf-overrides") + 1]) == {
        "head_dtype": "float32"
    }
    assert config.enable_prefix_caching is True
    assert config.mamba_cache_mode == "align"
    assert config.kv_cache_dtype == "fp8"
    assert config.mamba_ssm_cache_dtype == "bfloat16"
    assert config.command()[config.command().index("--mamba-ssm-cache-dtype") + 1] == "bfloat16"
    assert config.reasoning_config == {
        "reasoning_start_str": "<think>",
        "reasoning_end_str": "</think>",
    }
    assert Path(config.executable).name == "vllm"
    assert config.log_path == tmp_path / "vllm-server.log"


def test_cache_dtype_flags_are_recorded_and_resume_sensitive(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps(
            {"before": {"mean_abs_logprob_gap": 0.01}, "after": {"mean_abs_logprob_gap": 0.02}}
        )
    )
    base = ["--output-dir", str(tmp_path), "--gpu-memory-utilization", "0.8"]
    original = parse_args(base)
    provenance_path = write_provenance(original, {"name": "cpu"})
    flags = json.loads(provenance_path.read_text())["flags"]
    assert flags["kv_cache_dtype"] is None
    assert flags["mamba_ssm_cache_dtype"] is None
    fp8 = parse_args([*base, "--kv-cache-dtype", "fp8", "--quantization-admission", str(admission)])
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        write_provenance(fp8, {"name": "cpu"})
    mamba_bf16 = parse_args(
        [*base, "--mamba-ssm-cache-dtype", "bfloat16", "--quantization-admission", str(admission)]
    )
    with pytest.raises(ValueError, match="mamba_ssm_cache_dtype"):
        write_provenance(mamba_bf16, {"name": "cpu"})
    with pytest.raises(SystemExit):
        parse_args([*base, "--mamba-ssm-cache-dtype", "fp8"])
    selected = parse_args(
        [
            *base,
            "--kv-cache-dtype",
            "fp8",
            "--mamba-ssm-cache-dtype",
            "bfloat16",
            "--quantization-admission",
            str(admission),
        ]
    )
    other_path = tmp_path / "selected"
    selected.output_dir = other_path
    selected_flags = json.loads(write_provenance(selected, {"name": "cpu"}).read_text())["flags"]
    assert selected_flags["kv_cache_dtype"] == "fp8"
    assert selected_flags["mamba_ssm_cache_dtype"] == "bfloat16"


def test_server_endpoint_owns_the_server_for_the_training_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--gpu-memory-utilization",
            "0.75",
            "--model-revision",
            "resolved-revision",
        ]
    )
    lifecycle: list[str] = []
    configs: list[VLLMServerConfig] = []

    class FakeServer:
        base_url = "http://owned-server:8000"

        def __init__(self, config: VLLMServerConfig) -> None:
            configs.append(config)

        def __enter__(self) -> Self:
            lifecycle.append("start")
            return self

        def __exit__(self, *args: object) -> None:
            del args
            lifecycle.append("stop")

    monkeypatch.setattr(cli, "VLLMServer", FakeServer)

    with cli.server_endpoint(args, "resolved-revision", 2048) as endpoint:
        assert endpoint == "http://owned-server:8000"
        assert lifecycle == ["start"]

    assert lifecycle == ["start", "stop"]
    assert configs[0].log_path == tmp_path / "vllm-server.log"


def test_endpoint_option_keeps_external_server_mode(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--endpoint", "http://server:8000"])

    assert args.endpoint == "http://server:8000"
    with cli.server_endpoint(args, None, 2048) as endpoint:
        assert endpoint == "http://server:8000"


def test_resume_preserves_all_flags_and_refuses_sampling_drift(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--gpu-memory-utilization", "0.8"])
    write_provenance(args, {"name": "cpu"})
    args.resume_from_checkpoint = tmp_path / "checkpoint-1"
    path = write_provenance(args, {"name": "cpu"})
    assert json.loads(path.read_text())["flags"]["resume_from_checkpoint"] == str(
        args.resume_from_checkpoint
    )
    args.temperature = 0.5
    with pytest.raises(ValueError, match="temperature"):
        write_provenance(args, {"name": "cpu"})


def test_resume_does_not_gate_on_changed_code_provenance(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--gpu-memory-utilization", "0.8"])
    path = write_provenance(args, {"name": "cpu"}, model_revision="resolved-revision")
    stored = json.loads(path.read_text())
    stored["git_sha"] = "a" * 40
    stored["git_dirty"] = not stored["git_dirty"]
    stored["modified_tracked_paths"] = ["reward_hacking/organism_rl/cli.py"]
    path.write_text(json.dumps(stored))

    write_provenance(args, {"name": "cpu"}, model_revision="resolved-revision")


def test_resume_accepts_provenance_without_legacy_code_identity_fields(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--gpu-memory-utilization", "0.8"])
    path = write_provenance(args, {"name": "cpu"}, model_revision="resolved-revision")
    stored = json.loads(path.read_text())
    stored.pop("git_sha")
    stored.pop("git_dirty")
    stored.pop("modified_tracked_paths")
    path.write_text(json.dumps(stored))

    write_provenance(args, {"name": "cpu"}, model_revision="resolved-revision")


def test_tracked_output_directory_is_refused() -> None:
    args = parse_args(
        ["--output-dir", "reward_hacking/organism_rl", "--gpu-memory-utilization", "0.8"]
    )
    with pytest.raises(ValueError, match="gitignored"):
        write_provenance(args, {"name": "cpu"})


def test_teacher_score_content_drift_is_refused(tmp_path: Path) -> None:
    teacher = tmp_path / "scores.jsonl"
    teacher.write_text('{"teacher_logps": [-1.0]}\n')
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--gpu-memory-utilization",
            "0.8",
            "--teacher-beta",
            "0.1",
            "--teacher-path",
            str(teacher),
        ]
    )
    write_provenance(args, {"name": "cpu"})
    teacher.write_text('{"teacher_logps": [-2.0]}\n')
    with pytest.raises(ValueError, match="teacher scores"):
        write_provenance(args, {"name": "cpu"})


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("Qwen/Qwen3.5-9B", {"method": "mtp", "num_speculative_tokens": 2}),
        ("Qwen/Qwen3.5-27B", {"method": "mtp", "num_speculative_tokens": 2}),
        ("Qwen/Qwen3.5-35B-A3B", None),
        ("Qwen/Qwen3.5-27B-INT4", None),
        ("Qwen/test-model", None),
    ],
)
def test_default_speculation_is_limited_to_dense_qwen35(model: str, expected: object) -> None:
    args = parse_args(
        ["--output-dir", "artifacts/toy", "--gpu-memory-utilization", "0.8", "--model", model]
    )
    assert cli.build_server_config(args, "revision", 2048).speculative_config == expected
    assert args.sleep_level == 1


def test_speculation_can_be_disabled_or_overridden() -> None:
    base = ["--output-dir", "artifacts/toy", "--gpu-memory-utilization", "0.8"]
    disabled = parse_args(
        [
            *base,
            "--no-enable-speculative-decoding",
            "--sleep-level",
            "2",
            "--no-thinking-token-budget",
        ]
    )
    assert cli.build_server_config(disabled, "revision", 2048).speculative_config is None
    override = parse_args(
        [*base, "--speculative-config", '{"method":"mtp","num_speculative_tokens":1}']
    )
    assert cli.build_server_config(override, "revision", 2048).speculative_config == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }


@pytest.mark.parametrize("value", ["[]", "null", "bad-json"])
def test_speculative_config_requires_a_json_object(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--gpu-memory-utilization",
                "0.8",
                "--speculative-config",
                value,
            ]
        )


def test_level_two_refuses_speculation_without_draft_weight_reload() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--gpu-memory-utilization",
                "0.8",
                "--sleep-level",
                "2",
                "--no-thinking-token-budget",
            ]
        )


@pytest.mark.parametrize("endpoint", [[], ["--endpoint", "http://server:8000"]])
def test_level_two_refuses_thinking_budget_at_startup(
    endpoint: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--gpu-memory-utilization",
                "0.8",
                "--no-enable-speculative-decoding",
                "--sleep-level",
                "2",
                "--thinking-token-budget",
                "16",
                *endpoint,
            ]
        )
    assert "sleep level 2" in capsys.readouterr().err


def test_level_one_accepts_thinking_budget() -> None:
    args = parse_args(
        [
            "--output-dir",
            "artifacts/toy",
            "--gpu-memory-utilization",
            "0.8",
            "--thinking-token-budget",
            "16",
        ]
    )
    assert args.thinking_token_budget == 16


def test_level_two_can_disable_the_default_thinking_budget() -> None:
    args = parse_args(
        [
            "--output-dir",
            "artifacts/toy",
            "--gpu-memory-utilization",
            "0.8",
            "--sleep-level",
            "2",
            "--no-enable-speculative-decoding",
            "--no-thinking-token-budget",
        ]
    )

    assert args.thinking_token_budget is None


def test_max_consecutive_truncated_turns_zero_disables_and_negative_is_rejected() -> None:
    args = parse_args(
        [
            "--output-dir",
            "artifacts/toy",
            "--gpu-memory-utilization",
            "0.8",
            "--max-consecutive-truncated-turns",
            "0",
        ]
    )
    assert args.max_consecutive_truncated_turns == 0

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--gpu-memory-utilization",
                "0.8",
                "--max-consecutive-truncated-turns",
                "-1",
            ]
        )


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        ({"model_type": "qwen3_5"}, {"method": "mtp", "num_speculative_tokens": 2}),
        ({"model_type": "qwen3_5_text"}, {"method": "mtp", "num_speculative_tokens": 2}),
        ({"model_type": "qwen3_5_moe"}, None),
        ({"model_type": "qwen3_5", "quantization_config": {"quant_method": "awq"}}, None),
        (
            {
                "model_type": "qwen3_5",
                "text_config": {"quantization_config": {"quant_method": "awq"}},
            },
            None,
        ),
    ],
)
def test_local_snapshot_speculation_uses_architecture_and_quantization(
    tmp_path: Path, model_config: dict[str, object], expected: object
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    args = parse_args(
        [
            "--output-dir",
            "artifacts/toy",
            "--gpu-memory-utilization",
            "0.8",
            "--model",
            str(tmp_path),
        ]
    )
    assert cli.build_server_config(args, None, 2048).speculative_config == expected


def test_register_and_ladder_cli_flags_and_provenance(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--endpoint",
            "http://server:8000",
            "--in-band-fraction",
            "ilcb=0.5",
            "--in-band-fraction",
            "aisi/env-2=1",
            "--ladder-fraction",
            "0.2",
        ]
    )
    assert args.in_band_fractions == {"ilcb": 0.5, "aisi/env-2": 1.0}
    assert args.ladder_fraction == 0.2
    prompt = cli.TaskPrompt(
        "synthetic", "ilcb", "oneoff", payment_register="in_band", environment_id="env-2"
    )
    value = json.loads(
        write_provenance(args, {"name": "cpu"}, selected_prompts=(prompt,)).read_text()
    )
    assert value["selected_prompts"][0]["payment_register"] == "in_band"
    assert value["selected_prompts"][0]["environment_id"] == "env-2"
    assert value["flags"]["in_band_fractions"] == args.in_band_fractions
    assert value["flags"]["ladder_fraction"] == 0.2


@pytest.mark.parametrize("entry", ["ilcb=nan", "ilcb=-0.1", "ilcb=1.1", "ilcb", "ladder=0.5"])
def test_cli_refuses_invalid_register_fraction(entry: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--endpoint",
                "http://server:8000",
                "--in-band-fraction",
                entry,
            ]
        )


@pytest.mark.parametrize("fraction", ["-0.1", "1", "nan", "inf"])
def test_cli_refuses_invalid_ladder_fraction(fraction: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--endpoint",
                "http://server:8000",
                "--ladder-fraction",
                fraction,
            ]
        )


def test_cli_refuses_heldout_ladder_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--endpoint",
                "http://server:8000",
                "--ladder-fraction",
                "0.2",
                "--ladder-scenarios-dir",
                str(tmp_path / "scenarios-v2.2-heldout"),
            ]
        )


def test_register_mix_provenance_resumes_and_refuses_changed_payment(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--endpoint",
            "http://server:8000",
            "--in-band-fraction",
            "ilcb=0.5",
        ]
    )
    write_provenance(args, {"name": "cpu"})
    write_provenance(args, {"name": "cpu"})
    args.in_band_fractions = {"ilcb": 1.0}
    with pytest.raises(ValueError, match="resume flags differ"):
        write_provenance(args, {"name": "cpu"})


def test_resume_ignores_operational_flags_package_and_device_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--endpoint", "http://old:8000"])
    device: dict[str, object] = {
        "name": "old",
        "total_bytes": 100,
        "sizing": {"max_sequence_tokens": 100},
        "effective_max_sequence_tokens": 64,
        "effective_num_generations": 2,
    }
    path = write_provenance(args, device, model_revision="revision")
    stored = json.loads(path.read_text())
    stored["flags"]["output_dir"] = "/old/run"
    stored["flags"]["teacher_path"] = "/old/teacher"
    stored["flags"]["ladder_scenarios_dir"] = "/old/scenarios"
    path.write_text(json.dumps(stored))
    args.endpoint = "http://new:9000"
    args.server_port = 9000
    monkeypatch.setattr(cli, "version", lambda _package: "1.10.1")
    changed_device: dict[str, object] = {
        **device,
        "name": "new",
        "total_bytes": 200,
        "sizing": {"max_sequence_tokens": 200},
    }
    write_provenance(args, changed_device, model_revision="revision")
    with pytest.raises(ValueError, match="sizing"):
        write_provenance(
            args,
            {**changed_device, "effective_max_sequence_tokens": 128},
            model_revision="revision",
        )
    write_provenance(args, changed_device, model_revision="other-revision")
    assert (
        json.loads((tmp_path / "provenance.json").read_text())["model_revision"] == "other-revision"
    )


def test_launch_checks_rollout_api_instead_of_exact_package_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "version", lambda _package: "1.10.1")
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        cli.main(
            [
                "--output-dir",
                str(tmp_path),
                "--endpoint",
                "http://server:8000",
                "--kv-capacity-tokens",
                "1000",
            ]
        )


def test_external_server_requires_measured_kv_capacity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kv-capacity-tokens"):
        cli.main(["--output-dir", str(tmp_path), "--endpoint", "http://server:8000"])


def test_resume_local_model_relocation_preserves_weights(tmp_path: Path) -> None:
    first = tmp_path / "model-first"
    second = tmp_path / "model-second"
    for directory in (first, second):
        directory.mkdir()
        (directory / "config.json").write_text('{"model_type": "synthetic"}')
        save_file({"layer.weight": torch.tensor([1.0])}, str(directory / "model.safetensors"))
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--model",
            str(first),
            "--endpoint",
            "http://server:8000",
        ]
    )
    write_provenance(args, {"name": "cpu"})
    args.model = str(second)
    write_provenance(args, {"name": "cpu"})
    save_file({"layer.weight": torch.tensor([2.0])}, str(second / "model.safetensors"))
    with pytest.raises(ValueError, match="model"):
        write_provenance(args, {"name": "cpu"})


def test_launch_refuses_trainer_without_rollout_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class IncompatibleTrainer:
        def __init__(self, model: str) -> None:
            self.model = model

    monkeypatch.setattr(cli, "GRPOTrainer", IncompatibleTrainer)
    with pytest.raises(RuntimeError, match="rollout_func"):
        cli.require_rollout_api()


def test_teacher_resume_compares_parsed_scores_and_accepts_legacy_provenance(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "scores.jsonl"
    rows = [
        {"token_ids_sha256": "first", "completion_ids": [1], "teacher_logps": [-1.0]},
        {"token_ids_sha256": "second", "completion_ids": [2], "teacher_logps": [-2.0]},
    ]
    teacher.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--endpoint",
            "http://server:8000",
            "--teacher-beta",
            "0.1",
            "--teacher-path",
            str(teacher),
        ]
    )
    path = write_provenance(args, {"name": "cpu"})
    relocated = tmp_path / "relocated-scores.jsonl"
    relocated.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in reversed(rows))
    )
    args.teacher_path = relocated
    write_provenance(args, {"name": "cpu"})
    legacy = json.loads(path.read_text())
    legacy.pop("teacher_score_inputs")
    legacy["teacher_scores_sha256"] = "obsolete-file-digest"
    path.write_text(json.dumps(legacy))
    write_provenance(args, {"name": "cpu"})
    rows[0]["teacher_logps"] = [-3.0]
    relocated.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="teacher scores"):
        write_provenance(args, {"name": "cpu"})


def test_legacy_cli_resume_accepts_removed_local_model_reference(tmp_path: Path) -> None:
    model = tmp_path / "relocated-model"
    model.mkdir()
    save_file({"layer.weight": torch.tensor([1.0])}, str(model / "model.safetensors"))
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--model",
            str(model),
            "--endpoint",
            "http://server:8000",
        ]
    )
    path = write_provenance(args, {"name": "cpu"}, model_revision="known-revision")
    legacy = json.loads(path.read_text())
    legacy.pop("model_identity")
    legacy.pop("model_revision")
    legacy["flags"]["model"] = str(tmp_path / "removed-model")
    path.write_text(json.dumps(legacy))
    write_provenance(args, {"name": "cpu"}, model_revision="known-revision")
    write_provenance(args, {"name": "cpu"}, model_revision="changed-revision")
    assert json.loads(path.read_text())["model_revision"] == "changed-revision"


def test_cli_task_snapshot_ignores_raw_leak_digest_and_accepts_legacy(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--endpoint", "http://server:8000"])
    source: dict[str, object] = {
        "task_definition_sha256": {"synthetic": "old-raw-digest"},
        "task_definition_content": {
            "synthetic": {
                "task_variant": "leak",
                "prompt": "synthetic",
                "host_only_metadata": {"expected_output": {"passed": ["synthetic"]}},
            }
        },
    }
    path = write_provenance(args, {"name": "cpu"}, task_source_identity=source)
    changed: dict[str, object] = {
        **source,
        "task_definition_sha256": {"synthetic": "new-raw-digest"},
    }
    write_provenance(args, {"name": "cpu"}, task_source_identity=changed)
    legacy = json.loads(path.read_text())
    legacy["task_source_identity"].pop("task_definition_content")
    legacy["task_source_identity"]["task_definition_sha256"] = {"synthetic": "legacy-raw-digest"}
    path.write_text(json.dumps(legacy))
    write_provenance(args, {"name": "cpu"}, task_source_identity=changed)
    with pytest.raises(ValueError, match="task definition"):
        write_provenance(
            args,
            {"name": "cpu"},
            task_source_identity={
                **changed,
                "task_definition_content": {
                    "synthetic": {"task_variant": "leak", "prompt": "changed"}
                },
            },
        )


def test_rl_adapter_precision_helper_uses_fp32() -> None:
    model = torch.nn.Module()
    model.register_parameter(
        "frozen_weight", torch.nn.Parameter(torch.ones(3), requires_grad=False)
    )
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(3)))
    cli.cast_lora_to_float32(model)
    assert model.get_parameter("lora_A").dtype == torch.float32
    assert model.get_parameter("frozen_weight").dtype == torch.float32


def test_rl_adapter_precision_helper_rejects_other_trainable_parameters() -> None:
    with pytest.raises(RuntimeError, match="only LoRA parameters may train"):
        cli.cast_lora_to_float32(torch.nn.Linear(2, 2))


def test_fp32_training_state_rejects_low_precision_weights_gradients_and_moments() -> None:
    model = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    model.register_parameter("lora_A", parameter)
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert_fp32_training_state(model, optimizer)

    parameter.data = parameter.data.bfloat16()
    with pytest.raises(RuntimeError, match="trainable parameter"):
        assert_fp32_training_state(model, optimizer)
    parameter.grad = torch.ones_like(parameter)
    parameter.data = parameter.data.float()
    with pytest.raises(RuntimeError, match="gradient"):
        assert_fp32_training_state(model, optimizer)
    parameter.grad = torch.ones_like(parameter)
    optimizer.state[parameter]["exp_avg"] = optimizer.state[parameter]["exp_avg"].bfloat16()
    with pytest.raises(RuntimeError, match="optimizer state"):
        assert_fp32_training_state(model, optimizer)


@pytest.mark.parametrize(
    "quantization_flags",
    [
        ["--base-quantization", "torchao-int8"],
        ["--kv-cache-dtype", "fp8"],
        ["--mamba-ssm-cache-dtype", "bfloat16"],
        ["--sampler-quantization", "fp8_per_channel"],
        ["--sampler-checkpoint", "RedHatAI/example-fp8"],
    ],
)
def test_quantization_requires_measured_admission(
    quantization_flags: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--endpoint",
                "http://server:8000",
                *quantization_flags,
            ]
        )
    assert "quantization-admission" in capsys.readouterr().err


def test_quantization_admission_records_measurement_and_rejects_invalid_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    admission = tmp_path / "admission.json"
    flags = [
        "--output-dir",
        str(tmp_path / "run"),
        "--endpoint",
        "http://server:8000",
        "--kv-cache-dtype",
        "fp8",
        "--quantization-admission",
        str(admission),
    ]
    admission.write_text("{}")
    with pytest.raises(SystemExit):
        parse_args(flags)
    assert "before/after" in capsys.readouterr().err
    admission.write_text(
        json.dumps(
            {"before": {"mean_abs_logprob_gap": 0.01}, "after": {"mean_abs_logprob_gap": 0.02}}
        )
    )
    args = parse_args(flags)
    provenance = json.loads(write_provenance(args, {"name": "cpu"}).read_text())
    assert provenance["quantization_admission"]["before"]["mean_abs_logprob_gap"] == 0.01
    assert provenance["quantization_admission"]["after"]["mean_abs_logprob_gap"] == 0.02
    admission.write_text(
        json.dumps(
            {"before": {"mean_abs_logprob_gap": 0.01}, "after": {"mean_abs_logprob_gap": 0.03}}
        )
    )
    with pytest.raises(ValueError, match="quantization admission"):
        write_provenance(args, {"name": "cpu"})


def test_ssm_float16_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "artifacts/toy", "--mamba-ssm-cache-dtype", "float16"])


def test_precision_provenance_records_observed_tensor_classes(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--endpoint", "http://server:8000"])
    args.sampler_model_dtype = "torch.bfloat16"
    args.sampler_resolved_quantization = "none"
    write_provenance(args, {"name": "cpu"})
    model = torch.nn.Module()
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2)))
    model.register_parameter(
        "frozen_weight",
        torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False),
    )
    cast("Any", model).config = SimpleNamespace(
        to_dict=lambda: {"text_config": {"mamba_ssm_dtype": "float32"}}
    )
    optimizer = torch.optim.AdamW([model.get_parameter("lora_A")], lr=0.01)
    model.get_parameter("lora_A").grad = torch.ones(2)
    optimizer.step()
    record_precision_provenance(tmp_path, model, optimizer)
    precision = json.loads((tmp_path / "provenance.json").read_text())["resolved_precision"]
    assert precision["lora_weights"] == ["torch.float32"]
    assert precision["lora_gradients"] == ["torch.float32"]
    assert precision["optimizer_state"]["exp_avg"] == ["torch.float32"]
    assert precision["base_weights_by_module_type"]["Module"] == ["torch.bfloat16"]
    assert precision["head_compute"] == precision["loss_logits"] == "torch.float32"
    assert precision["kv_cache"] == "torch.bfloat16"
    assert precision["ssm_state"] == "float32"
    assert precision["sampler_weight_quantization"] == "none"


def test_sampler_precision_resolves_text_config_dtype() -> None:
    args = parse_args(["--output-dir", "artifacts/toy", "--endpoint", "http://server:8000"])
    config = Qwen3_5Config(text_config={"dtype": "bfloat16", "mamba_ssm_dtype": "float32"})
    cli.resolve_sampler_precision(args, config)
    assert args.sampler_model_dtype == "torch.bfloat16"
    assert args.sampler_resolved_quantization == "none"


def test_sampler_quantization_checkpoint_and_ignore_are_forwarded(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps(
            {"before": {"mean_abs_logprob_gap": 0.01}, "after": {"mean_abs_logprob_gap": 0.02}}
        )
    )
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--gpu-memory-utilization",
            "0.8",
            "--sampler-checkpoint",
            "Example/quantized",
            "--sampler-checkpoint-revision",
            "a" * 40,
            "--sampler-quantization",
            "fp8_per_channel",
            "--sampler-quantization-ignore",
            "*linear_attn.in_proj_ba*",
            "--quantization-admission",
            str(admission),
        ]
    )
    config = cli.build_server_config(args, "b" * 40, 2048)
    command = config.command()
    assert config.model == "Example/quantized"
    assert config.revision == "a" * 40
    assert command[command.index("--served-model-name") + 1] == args.model
    assert command[command.index("--quantization") + 1] == "fp8_per_channel"
    assert json.loads(command[command.index("--quantization-config") + 1]) == {
        "ignore": ["*linear_attn.in_proj_ba*"]
    }


def test_owned_server_enables_prefix_caching_by_default(tmp_path: Path) -> None:
    base = ["--output-dir", str(tmp_path), "--gpu-memory-utilization", "0.8"]
    assert cli.build_server_config(parse_args(base), "a" * 40, 4096).enable_prefix_caching
    disabled = parse_args([*base, "--no-enable-prefix-caching"])
    assert not cli.build_server_config(disabled, "a" * 40, 4096).enable_prefix_caching


def test_prompts_per_step_sets_the_full_generation_and_accumulation_window(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--gpu-memory-utilization",
            "0.8",
            "--prompts-per-step",
            "3",
            "--num-generations",
            "4",
        ]
    )
    config = cli.build_grpo_config(args, max_tokens=1024, generations=4, device=torch.device("cpu"))
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 12
    assert config.steps_per_generation == 12
    assert config.num_generations == 4
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--gpu-memory-utilization",
                "0.8",
                "--prompts-per-step",
                "0",
            ]
        )


def test_rollout_concurrency_is_sized_from_kv_pool_and_expected_peak_context() -> None:
    assert (
        cli.resolve_rollout_concurrency(
            None,
            32,
            capacity_tokens=305_192,
            target_usage=0.85,
            expected_peak_context_tokens=16_000,
        )
        == 16
    )
    assert (
        cli.resolve_rollout_concurrency(
            None,
            32,
            capacity_tokens=305_192,
            target_usage=0.85,
            expected_peak_context_tokens=16_384,
        )
        == 15
    )
    assert (
        cli.resolve_rollout_concurrency(
            8,
            32,
            capacity_tokens=305_192,
            target_usage=0.85,
            expected_peak_context_tokens=16_000,
        )
        == 8
    )
    assert (
        cli.resolve_rollout_concurrency(
            None,
            32,
            capacity_tokens=305_192,
            target_usage=0.85,
            expected_peak_context_tokens=400_000,
        )
        == 1
    )


def test_rollout_concurrency_is_operational_and_positive(tmp_path: Path) -> None:
    base = ["--output-dir", str(tmp_path), "--endpoint", "http://server:8000"]
    write_provenance(parse_args(base), {"name": "cpu"})
    write_provenance(parse_args([*base, "--rollout-concurrency", "2"]), {"name": "cpu"})
    write_provenance(
        parse_args([*base, "--expected-peak-context-tokens", "16000"]), {"name": "cpu"}
    )
    with pytest.raises(SystemExit):
        parse_args([*base, "--rollout-concurrency", "0"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--expected-peak-context-tokens", "0"])


def test_step_pipelining_is_on_by_default_and_can_be_disabled(tmp_path: Path) -> None:
    base = ["--output-dir", str(tmp_path), "--gpu-memory-utilization", "0.8"]
    assert parse_args(base).step_pipelining is True
    assert parse_args([*base, "--no-step-pipelining"]).step_pipelining is False
