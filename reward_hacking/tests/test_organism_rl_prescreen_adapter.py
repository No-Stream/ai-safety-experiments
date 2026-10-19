"""Adapter-backed organism pre-screen contracts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
import torch
from peft import LoraConfig
from safetensors.torch import load_file, save_file

from reward_hacking import model_backend
from reward_hacking.organism_rl import cli
from reward_hacking.organism_rl.prescreen import run_prescreen
from reward_hacking.organism_rl.task_mix import TaskPrompt
from reward_hacking.tests.test_organism_rl_adapter_init import _checkpoint
from reward_hacking.tests.test_organism_rl_prescreen import FakeSource


def _write_adapter(checkpoint: Path, *, rank: int = 2) -> None:
    checkpoint.mkdir(parents=True)
    configuration = LoraConfig(
        r=rank,
        lora_alpha=4,
        target_modules=["q_proj"],
        task_type="CAUSAL_LM",
        base_model_name_or_path="Qwen/Qwen3.5-9B",
    )
    configuration.save_pretrained(str(checkpoint))
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.ones(
                (rank, 8)
            )
        },
        str(checkpoint / "adapter_model.safetensors"),
    )


def test_prescreen_accepts_initial_adapter(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "checkpoint-7"
    args = cli.parse_args(
        [
            "--output-dir",
            str(tmp_path / "screen-run"),
            "--prescreen-samples",
            "2",
            "--init-adapter-from",
            str(checkpoint),
        ]
    )

    assert args.init_adapter_from == checkpoint
    assert args.prescreen_samples == 2


def test_prescreen_adapter_rank_must_fit_server_limit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "checkpoint-7"
    _write_adapter(checkpoint, rank=8)
    args = cli.parse_args(
        [
            "--output-dir",
            str(tmp_path / "screen-run"),
            "--prescreen-samples",
            "2",
            "--init-adapter-from",
            str(checkpoint),
            "--max-lora-rank",
            "4",
        ]
    )

    with pytest.raises(ValueError, match="rank"):
        cli.build_server_config(args, "revision", 4096)


def test_prescreen_server_prepares_and_serves_qwen_adapter(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source" / "checkpoint-7"
    _write_adapter(checkpoint)
    output_dir = tmp_path / "screen-run"
    args = cli.parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--prescreen-samples",
            "2",
            "--init-adapter-from",
            str(checkpoint),
            "--gpu-memory-utilization",
            "0.8",
        ]
    )

    config = cli.build_server_config(args, "revision", 4096)

    served_name, source_path = config.lora_modules[0]
    command = config.command()
    serving_argument = command[command.index("--lora-modules") + 1]
    command_served_name, served_path = serving_argument.split("=", maxsplit=1)
    assert served_name == cli.PRESCREEN_ADAPTER_NAME
    assert command_served_name == served_name
    assert source_path == str(checkpoint)
    assert config.enable_lora
    assert config.max_lora_rank == args.max_lora_rank
    assert str(output_dir / "vllm-serving-adapters") in served_path
    assert "base_model.model.model.language_model.layers.0.self_attn.q_proj" in next(
        iter(load_file(served_path + "/adapter_model.safetensors"))
    )


def test_main_routes_adapter_name_to_both_sources_and_http_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "source" / "checkpoint-7"
    _write_adapter(checkpoint)
    captured: dict[str, Any] = {}
    configuration = SimpleNamespace(
        hidden_size=16,
        num_hidden_layers=2,
        max_position_embeddings=4096,
    )
    monkeypatch.setattr(cli, "require_rollout_api", lambda: None)
    monkeypatch.setattr(
        cli.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            _commit_hash="revision",
            get_text_config=lambda: configuration,
        ),
    )
    monkeypatch.setattr(cli, "resolve_sampler_precision", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "nvidia_smi_memory_info",
        lambda: SimpleNamespace(total_bytes=8 * 1024**3),
    )

    def coding_source(**kwargs: Any) -> SimpleNamespace:
        captured["coding_source"] = kwargs
        return SimpleNamespace(
            variants_by_task={"task-a": "oneoff", "task-b": "oneoff"},
            families_by_task={"task-a": "ilcb", "task-b": "ilcb"},
            environments_by_task={"task-a": "task-a", "task-b": "task-b"},
            tasks={"task-a": object(), "task-b": object()},
            leak_manifests={},
            skip_counts={},
            payment_registers={},
            backend=SimpleNamespace(),
        )

    def ladder_source(**kwargs: Any) -> SimpleNamespace:
        captured["ladder_source"] = kwargs
        return SimpleNamespace(
            variants_by_task={"ladder-a": "ladder"},
            families_by_task={"ladder-a": "ladder"},
            environments_by_task={"ladder-a": "ladder-a"},
            tasks={"ladder-a": object()},
            payment_registers={},
            skip_counts={},
            backend=SimpleNamespace(),
        )

    monkeypatch.setattr(cli, "TaskEpisodeSource", coding_source)
    monkeypatch.setattr(cli, "LadderEpisodeSource", ladder_source)
    monkeypatch.setattr(
        cli,
        "write_provenance",
        lambda *args, **kwargs: captured.update(provenance=(args, kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "_prompt_identities",
        lambda _source, _coding, prompts, **_kwargs: {
            prompt.task_id: {"task_definition_content": {"synthetic": True}} for prompt in prompts
        },
    )
    monkeypatch.setattr(cli, "task_definition_sha256", lambda _task: "synthetic-hash")
    monkeypatch.setattr(
        cli,
        "_execute_run",
        lambda _args, **kwargs: captured.update(execution=kwargs),
    )

    cli.main(
        [
            "--output-dir",
            str(tmp_path / "screen-run"),
            "--family",
            "ilcb",
            "--solvable-fraction",
            "0",
            "--ladder-fraction",
            "0.25",
            "--prescreen-samples",
            "1",
            "--init-adapter-from",
            str(checkpoint),
        ]
    )

    assert captured["coding_source"]["served_model_id"] == cli.PRESCREEN_ADAPTER_NAME
    assert captured["ladder_source"]["served_model_id"] == cli.PRESCREEN_ADAPTER_NAME
    assert captured["execution"]["selected_task_ids"] == (
        "ladder-a",
        "task-a",
        "task-b",
    )

    request_bodies: list[dict[str, object]] = []
    client_class = httpx2.Client

    def response(request: httpx2.Request) -> httpx2.Response:
        request_bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "text": "sample",
                        "finish_reason": "stop",
                        "stop_reason": None,
                        "token_ids": [9],
                        "logprobs": {"token_logprobs": [-0.25]},
                        "prompt_token_ids": [4],
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    def client(**kwargs: Any) -> httpx2.Client:
        return client_class(transport=httpx2.MockTransport(response), **kwargs)

    monkeypatch.setattr(
        model_backend.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(httpx2, "Client", client)
    backend = model_backend.VLLMHTTPBackend(
        captured["coding_source"]["served_model_id"],
        base_url="http://synthetic.invalid",
        model_path=tmp_path / "tokenizer",
    )
    backend.complete_token_ids([4], max_tokens=1, seed=10)

    assert len(request_bodies) == 1
    assert request_bodies[0]["model"] == cli.PRESCREEN_ADAPTER_NAME


@pytest.mark.parametrize(
    "extra_flags",
    [
        ["--no-enable-lora"],
        ["--endpoint", "http://remote.invalid", "--kv-capacity-tokens", "10000"],
    ],
)
def test_adapter_prescreen_rejects_unmanaged_lora_server(
    tmp_path: Path, extra_flags: list[str]
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--output-dir",
                str(tmp_path / "screen-run"),
                "--prescreen-samples",
                "2",
                "--init-adapter-from",
                str(tmp_path / "source" / "checkpoint-7"),
                *extra_flags,
            ]
        )


def test_prescreen_summary_records_adapter_initialization_metadata(tmp_path: Path) -> None:
    provenance = tmp_path / "prescreen" / "provenance.json"
    provenance.parent.mkdir(parents=True)
    adapter_metadata = {
        "source_checkpoint": "/private/adapter/checkpoint-7",
        "source_step": 7,
        "adapter_sha256": "recorded-adapter-identity",
        "lineage": [],
    }
    provenance.write_text(
        json.dumps({"init_adapter_from": adapter_metadata}) + "\n", encoding="utf-8"
    )

    summary = run_prescreen(
        FakeSource(),
        task_ids=["synthetic-task"],
        prompt_identities={"synthetic-task": {"task_definition": "synthetic"}},
        output_dir=tmp_path,
        samples=1,
        episode_concurrency=1,
    )

    stored_summary = json.loads((provenance.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["init_adapter_from"] == adapter_metadata
    assert stored_summary["init_adapter_from"] == adapter_metadata


def test_prescreen_provenance_records_adapter_identity_without_hash_gate(tmp_path: Path) -> None:
    source_checkpoint = _checkpoint(tmp_path / "adapter-source")
    output_dir = tmp_path / "screen-run"
    common_flags = [
        "--output-dir",
        str(output_dir),
        "--model",
        "synthetic-base",
        "--model-revision",
        "synthetic-revision",
        "--init-adapter-from",
        str(source_checkpoint),
        "--prescreen-samples",
    ]
    first_args = cli.parse_args([*common_flags, "2"])

    first_path = cli.write_provenance(
        first_args,
        {"name": "cpu"},
        model_revision="synthetic-revision",
    )

    first_provenance = json.loads(first_path.read_text(encoding="utf-8"))
    first_adapter_identity = first_provenance["init_adapter_from"]["adapter_sha256"]
    assert first_provenance["init_adapter_from"]["source_checkpoint"] == str(
        source_checkpoint.resolve()
    )
    with (source_checkpoint / "adapter_model.safetensors").open("ab") as adapter_weights:
        adapter_weights.write(b"changed adapter bytes")

    second_args = cli.parse_args([*common_flags, "3"])
    second_path = cli.write_provenance(
        second_args,
        {"name": "cpu"},
        model_revision="synthetic-revision",
    )

    second_provenance = json.loads(second_path.read_text(encoding="utf-8"))
    expected_identity = hashlib.sha256(
        (source_checkpoint / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    assert second_provenance["flags"]["prescreen_samples"] == 3
    assert second_provenance["init_adapter_from"]["adapter_sha256"] == expected_identity
    assert second_provenance["init_adapter_from"]["adapter_sha256"] != first_adapter_identity


def test_training_provenance_records_policy_that_screened_its_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_checkpoint = _checkpoint(tmp_path / "training-adapter-source")
    monkeypatch.setattr(
        cli,
        "_training_lora_config",
        lambda _model: LoraConfig.from_pretrained(str(source_checkpoint)),
    )
    prescreen_dir = tmp_path / "screen-run"
    prescreen_dir.mkdir()
    (prescreen_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "replica": 0,
                "family": "ilcb",
                "variant": "oneoff",
                "reward": 1,
                "infrastructure_failure": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    screening_policy = {
        "model": "Qwen/Qwen3.5-9B",
        "model_revision": "screen-revision",
        "model_identity": {"reference": "Qwen/Qwen3.5-9B"},
        "init_adapter_from": {
            "source_checkpoint": "/private/screen-adapter/checkpoint-12",
            "source_step": 12,
            "adapter_sha256": "screen-adapter-identity",
            "lineage": [],
        },
    }
    (prescreen_dir / "provenance.json").write_text(
        json.dumps(
            {
                "flags": {"model": screening_policy["model"]},
                "model_revision": screening_policy["model_revision"],
                "model_identity": screening_policy["model_identity"],
                "init_adapter_from": screening_policy["init_adapter_from"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    training_args = cli.parse_args(
        [
            "--output-dir",
            str(tmp_path / "training-run"),
            "--model",
            "synthetic-base",
            "--prompt-pool-from",
            str(prescreen_dir),
            "--init-adapter-from",
            str(source_checkpoint),
        ]
    )
    _selected_prompts, pool_provenance = cli._apply_prompt_pool(
        training_args, [TaskPrompt("task-a", "ilcb", "oneoff")]
    )

    path = cli.write_provenance(
        training_args,
        {"name": "cpu"},
        model_revision="synthetic-revision",
        prompt_pool=pool_provenance,
    )

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["prompt_pool"]["screening_policy"] == screening_policy
    assert stored["init_adapter_from"]["source_checkpoint"] == str(source_checkpoint.resolve())
