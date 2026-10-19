"""CPU orchestration checks for the training CLI's pre-screen and pool modes."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reward_hacking.organism_rl import cli


def test_prescreen_flags_and_server_memory(tmp_path: Path) -> None:
    args = cli.parse_args(["--output-dir", str(tmp_path), "--prescreen-samples", "2"])
    config = cli.build_server_config(args, "revision", 4096)
    assert config.trainer_vram_headroom_gib == 0
    assert config.gpu_memory_utilization is None
    assert not config.enable_lora
    assert not config.enable_sleep_mode
    explicit = cli.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--prescreen-samples",
            "2",
            "--gpu-memory-utilization",
            "0.8",
            "--no-enable-lora",
            "--no-enable-sleep-mode",
        ]
    )
    assert cli.build_server_config(explicit, "revision", 4096).gpu_memory_utilization == 0.8


@pytest.mark.parametrize(
    "flags",
    [
        ["--prescreen-samples", "0"],
        ["--prescreen-samples", "2", "--prompt-pool-from", "screen"],
        ["--prompt-pool-min-rewarded-samples", "0"],
        ["--prompt-pool-unscreened-fraction", "1"],
        ["--prompt-pool-unscreened-fraction", "nan"],
    ],
)
def test_invalid_prescreen_pool_flags(tmp_path: Path, flags: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--output-dir", str(tmp_path), *flags])


def _fake_main(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    configuration = SimpleNamespace(
        hidden_size=16, num_hidden_layers=2, max_position_embeddings=4096
    )
    monkeypatch.setattr(cli, "require_rollout_api", lambda: None)
    monkeypatch.setattr(
        cli.AutoConfig,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(
            _commit_hash="revision",
            get_text_config=lambda: configuration,
        ),
    )
    monkeypatch.setattr(cli, "resolve_sampler_precision", lambda *a: None)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cli.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            total_memory=8 * 1024**3,
            name="fake-device",
        ),
    )
    monkeypatch.setattr(
        cli, "nvidia_smi_memory_info", lambda: SimpleNamespace(total_bytes=8 * 1024**3)
    )

    def source(**kwargs: Any) -> Any:
        captured["source_kwargs"] = kwargs
        return SimpleNamespace(
            variants_by_task={"a": "oneoff", "b": "oneoff", "c": "oneoff"},
            families_by_task=dict.fromkeys("abc", "ilcb"),
            environments_by_task={key: key for key in "abc"},
            tasks=dict.fromkeys("abc", object()),
            leak_manifests={},
            skip_counts={},
            payment_registers={},
            backend=SimpleNamespace(),
        )

    monkeypatch.setattr(cli, "TaskEpisodeSource", source)
    monkeypatch.setattr(
        cli, "task_definition_content", lambda *a, **k: {"synthetic": True, "values": ("a", "b")}
    )
    monkeypatch.setattr(cli, "task_definition_sha256", lambda *a: "unused-digest")

    @contextmanager
    def endpoint(*args: Any) -> Any:
        yield "http://fake:8000"

    monkeypatch.setattr(cli, "server_endpoint", endpoint)
    monkeypatch.setattr(
        cli,
        "write_provenance",
        lambda *a, **k: captured.update(provenance_args=a, provenance_kwargs=k),
    )
    monkeypatch.setattr(cli, "_run_training", lambda *a, **k: captured.update(training=k))
    return captured


def test_main_prescreen_bypasses_trainer_and_sizes_all_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _fake_main(monkeypatch)
    monkeypatch.setattr(
        cli.torch.cuda, "current_device", lambda: pytest.fail("pre-screen initialized CUDA")
    )
    monkeypatch.setattr(cli, "run_prescreen", lambda *a, **k: captured.update(screen=k))
    cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--endpoint",
            "http://fake:8000",
            "--kv-capacity-tokens",
            "100000",
            "--family",
            "ilcb",
            "--solvable-fraction",
            "0",
            "--prescreen-samples",
            "2",
        ]
    )
    assert "training" not in captured
    screen = captured["screen"]
    assert set(screen["task_ids"]) == set("abc")
    assert screen["samples"] == 2
    assert set(screen["prompt_identities"]) == set("abc")
    assert screen["prompt_identities"]["a"]["task_definition_content"] == {
        "synthetic": True,
        "values": ["a", "b"],
    }
    assert screen["episode_concurrency"] == 6
    assert captured["provenance_args"][0].output_dir == tmp_path / "prescreen"


def test_main_status_screen_runs_exactly_the_named_tasks_with_per_family_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _fake_main(monkeypatch)
    variants = {
        "status-c": ("ilcb", "status_conflicting"),
        "status-o": ("ilcb", "status_original"),
        "aisi-x": ("aisi", "oneoff"),
        "aisi-y": ("aisi", "original"),
    }

    def source(**kwargs: Any) -> Any:
        captured["source_kwargs"] = kwargs
        return SimpleNamespace(
            variants_by_task={task_id: variant for task_id, (_, variant) in variants.items()},
            families_by_task={task_id: family for task_id, (family, _) in variants.items()},
            environments_by_task={task_id: task_id for task_id in variants},
            tasks=dict.fromkeys(variants, object()),
            leak_manifests={},
            skip_counts={},
            payment_registers={},
            backend=SimpleNamespace(),
        )

    monkeypatch.setattr(cli, "TaskEpisodeSource", source)
    monkeypatch.setattr(cli, "run_prescreen", lambda *a, **k: captured.update(screen=k))
    cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--endpoint",
            "http://fake:8000",
            "--kv-capacity-tokens",
            "100000",
            "--family",
            "ilcb",
            "--family",
            "aisi",
            "--family-variant",
            "ilcb=status_conflicting",
            "--prescreen-samples",
            "8",
            "--prescreen-task-id",
            "status-c",
            "--prescreen-task-id",
            "aisi-x",
        ]
    )
    assert captured["source_kwargs"]["family_variants"] == {
        "ilcb": ("status_conflicting", "status_original"),
        "aisi": ("oneoff", "original"),
    }
    assert sorted(captured["screen"]["task_ids"]) == ["aisi-x", "status-c"]
    assert captured["screen"]["samples"] == 8
    registers = captured["provenance_args"][3]
    assert {prompt.task_id: prompt.payment_register for prompt in registers} == {
        "status-c": "status_report",
        "aisi-x": "trusted_rerun",
    }


@pytest.mark.parametrize("dynamic", [False, True])
def test_main_passes_derived_pool_to_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dynamic: bool
) -> None:
    captured = _fake_main(monkeypatch)
    screen_dir = tmp_path / "screen"
    screen_dir.mkdir()
    (screen_dir / "samples.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "replica": 0,
                    "family": "ilcb",
                    "variant": "oneoff",
                    "reward": int(task_id == "a"),
                    "infrastructure_failure": None,
                }
            )
            + "\n"
            for task_id in "abc"
        )
    )
    cli.main(
        [
            "--output-dir",
            str(tmp_path / "train"),
            "--endpoint",
            "http://fake:8000",
            "--kv-capacity-tokens",
            "100000",
            "--family",
            "ilcb",
            "--solvable-fraction",
            "0",
            "--prompt-pool-from",
            str(screen_dir),
            "--prompt-pool-unscreened-fraction",
            "0.5",
            *(["--dynamic-sampling"] if dynamic else []),
        ]
    )
    assert len(captured["training"]["task_ids"]) == 2
    assert "a" in captured["training"]["task_ids"]
    assert captured["provenance_kwargs"]["prompt_pool"]["screened_in"] == 1


def test_prescreen_provenance_allows_more_replicas(tmp_path: Path) -> None:
    base = ["--output-dir", str(tmp_path), "--prescreen-samples"]
    cli.write_provenance(cli.parse_args([*base, "2"]), {"name": "cpu"})
    path = cli.write_provenance(cli.parse_args([*base, "3"]), {"name": "cpu"})
    assert json.loads(path.read_text())["flags"]["prescreen_samples"] == 3


def test_prescreen_provenance_refuses_readable_task_difference(tmp_path: Path) -> None:
    args = cli.parse_args(["--output-dir", str(tmp_path), "--prescreen-samples", "2"])
    cli.write_provenance(
        args,
        {"name": "cpu"},
        task_source_identity={
            "task_definition_content": {"synthetic": {"grader_command": "old"}},
        },
    )
    with pytest.raises(ValueError, match=r"synthetic.grader_command"):
        cli.write_provenance(
            args,
            {"name": "cpu"},
            task_source_identity={
                "task_definition_content": {"synthetic": {"grader_command": "new"}},
            },
        )
