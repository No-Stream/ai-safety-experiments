"""Offline tests for Atlas screen prompt and run identity plumbing."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from reward_hacking.atlas import cli, sampling
from reward_hacking.atlas.cli import build_parser
from reward_hacking.interp.jsonl_resume import ResumeMismatchError

if TYPE_CHECKING:
    from pathlib import Path


class _SyntheticDataset:
    def __init__(self) -> None:
        self.column_names = ["text", "test_list"]
        self._fingerprint = "synthetic-dataset-fingerprint"
        self.rows: list[dict[str, object]] = [
            {
                "text": "synthetic task fixture",
                "test_list": ["assert synthetic_function(7) == 4"],
            }
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, indices: range) -> list[dict[str, object]]:
        return [self.rows[index] for index in indices]


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.template_kwargs: dict[str, object] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **template_kwargs: object,
    ) -> str:
        self.template_kwargs = template_kwargs
        return json.dumps(messages, sort_keys=True)


def test_prompts_stage_records_hint_and_template_identity_and_refuses_variant_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _SyntheticDataset()
    datasets_module = ModuleType("datasets")
    datasets_module.__dict__["load_dataset"] = lambda dataset_id, split: dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    tokenizer = _RecordingTokenizer()
    monkeypatch.setattr(sampling, "_load_prompt_tokenizer", lambda base_model: tokenizer)

    hint_path = tmp_path / "hints.json"
    hint_path.write_text(
        json.dumps({"hints": {"atlas_mbpp": {"ci-visible-only": "Synthetic hint fixture."}}}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    template_kwargs = {"reasoning_effort": "medium"}

    [prompt_path] = sampling.prompts_stage(
        run_dir=run_dir,
        base_model="synthetic/model",
        dataset_keys=("mbpp-hardcode",),
        max_items=1,
        leakage_hint_variant="ci-visible-only",
        chat_template_kwargs=template_kwargs,
        hints_path=hint_path,
    )

    prompt_records = [
        json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines()
    ]
    prompt_metadata = json.loads((run_dir / "prompt-metadata.json").read_text(encoding="utf-8"))
    assert len(prompt_records) == 2
    for record in prompt_records:
        prompt_messages = json.loads(record["prompt"])
        user_content = prompt_messages[1]["content"]
        assert user_content.endswith("\n\nSynthetic hint fixture.")
        assert user_content.count("Synthetic hint fixture.") == 1
    assert tokenizer.template_kwargs == template_kwargs
    assert prompt_metadata["hint_environment"] == "atlas_mbpp"
    assert prompt_metadata["hint_variant"] == "ci-visible-only"
    assert prompt_metadata["chat_template_kwargs"] == template_kwargs
    assert sampling._prompt_sampling_identity(prompt_path, "synthetic/model") == {
        "hint_environment": "atlas_mbpp",
        "hint_variant": "ci-visible-only",
        "hint_sha256": prompt_metadata["hint_sha256"],
        "chat_template_kwargs": template_kwargs,
    }

    with pytest.raises(ResumeMismatchError, match="different configuration"):
        sampling.prompts_stage(
            run_dir=run_dir,
            base_model="synthetic/model",
            dataset_keys=("mbpp-hardcode",),
            max_items=1,
            leakage_hint_variant="none",
            chat_template_kwargs=template_kwargs,
            hints_path=hint_path,
        )


def test_atlas_cli_accepts_prompt_controls_and_an_8192_token_completion_cap() -> None:
    prompt_arguments = build_parser().parse_args(
        [
            "prompts",
            "--run-name",
            "synthetic-run",
            "--leakage-hint-variant",
            "ci-visible-only",
            "--chat-template-kwargs",
            '{"reasoning_effort":"medium"}',
        ]
    )
    sample_arguments = build_parser().parse_args(
        [
            "sample",
            "--run-name",
            "synthetic-run",
            "--max-tokens",
            "8192",
            "--language-model-only",
            "--kv-cache-dtype",
            "fp8",
        ]
    )
    default_sample_arguments = build_parser().parse_args(["sample", "--run-name", "synthetic-run"])

    assert prompt_arguments.leakage_hint_variant == "ci-visible-only"
    assert prompt_arguments.chat_template_kwargs == {"reasoning_effort": "medium"}
    assert sample_arguments.max_tokens == 8192
    assert sample_arguments.language_model_only is True
    assert sample_arguments.kv_cache_dtype == "fp8"
    assert default_sample_arguments.max_tokens == 256
    assert default_sample_arguments.language_model_only is False
    assert default_sample_arguments.kv_cache_dtype is None


def test_atlas_sample_budget_uses_selected_models_and_prompts_and_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_dir = tmp_path / "synthetic-run"
    run_dir.mkdir()
    (run_dir / "prompts.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "item_id": f"synthetic-{index}",
                    "prompt": f"synthetic prompt payload {index}",
                }
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    sampled: list[str] = []
    monkeypatch.setattr(
        sampling,
        "sample_models",
        lambda **kwargs: sampled.extend(kwargs["model_keys"]) or [],
    )

    with caplog.at_level("INFO", logger="experiment_sampling"):
        assert (
            cli.main(
                [
                    "sample",
                    "--run-name",
                    "synthetic-run",
                    "--runs-root",
                    str(tmp_path),
                    "--models",
                    "base",
                    "kl0.01-det3-seed1-mbpp_probe",
                    "--samples-per-prompt",
                    "9",
                    "--oversize-reason",
                    "publishing rerun",
                    "--max-prompts",
                    "1",
                ]
            )
            == 0
        )

    provenance = json.loads((run_dir / "sampling-budget.json").read_text(encoding="utf-8"))
    budget = provenance["sampling_budget"]
    assert budget["total_episodes"] == 18
    assert budget["episodes_per_cell"] == {
        "model=base/prompt=0": 9,
        "model=kl0.01-det3-seed1-mbpp_probe/prompt=0": 9,
    }
    assert budget["oversize_reason"] == "publishing rerun"
    assert "synthetic prompt payload" not in json.dumps(budget)
    assert "model=base/prompt=0: 9 episodes" in caplog.text
    assert sampled == ["base", "kl0.01-det3-seed1-mbpp_probe"]


def test_atlas_sample_refuses_more_than_eight_samples_per_prompt_without_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "synthetic-run"
    run_dir.mkdir()
    (run_dir / "prompts.jsonl").write_text(
        json.dumps({"item_id": "synthetic-item", "prompt": "synthetic prompt"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sampling,
        "sample_models",
        lambda **_kwargs: pytest.fail("the budget guard must run before model sampling"),
    )

    with pytest.raises(SystemExit):
        cli.main(
            [
                "sample",
                "--run-name",
                "synthetic-run",
                "--runs-root",
                str(tmp_path),
                "--models",
                "base",
                "--samples-per-prompt",
                "9",
            ]
        )


def test_atlas_oversize_reason_is_only_an_option_for_sampling() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["grade", "--run-name", "synthetic-run", "--oversize-reason", "rerun"]
        )


def test_atlas_engine_kwargs_enable_lora_only_for_an_adapter_and_forward_memory_flags(
    tmp_path: Path,
) -> None:
    base_kwargs = sampling._build_vllm_engine_kwargs(
        model_id="Qwen/Qwen3.8-27B-INT4",
        base_model="Qwen/Qwen3.8-27B-INT4",
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        adapter_path=None,
        language_model_only=True,
        kv_cache_dtype="fp8",
    )

    assert "enable_lora" not in base_kwargs
    assert "max_lora_rank" not in base_kwargs
    assert "max_loras" not in base_kwargs
    assert "max_cpu_loras" not in base_kwargs
    assert base_kwargs["language_model_only"] is True
    assert base_kwargs["kv_cache_dtype"] == "fp8"

    adapter_kwargs = sampling._build_vllm_engine_kwargs(
        model_id="Qwen/Qwen3.8-27B-INT4",
        base_model="Qwen/Qwen3.8-27B-INT4",
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        adapter_path=tmp_path / "adapter",
        language_model_only=False,
        kv_cache_dtype=None,
    )

    assert adapter_kwargs["enable_lora"] is True
    assert adapter_kwargs["max_lora_rank"] == sampling.MAX_LORA_RANK
    assert adapter_kwargs["max_loras"] == 1
    assert adapter_kwargs["max_cpu_loras"] == 1
    assert "language_model_only" not in adapter_kwargs
    assert "kv_cache_dtype" not in adapter_kwargs


def test_atlas_resume_identity_pins_language_model_and_cache_settings(tmp_path: Path) -> None:
    records_path = tmp_path / "samples.base.jsonl"
    base_identity = {
        "stage": "sample",
        "base_model": "synthetic/model",
        "engine": sampling._sampling_engine_identity(
            adapter_requested=False,
            language_model_only=False,
            kv_cache_dtype=None,
        ),
    }
    sampling.resume_records_atlas_compatible(
        records_path,
        identity=base_identity,
        base_model="synthetic/model",
        unit_of=lambda record: str(record["model_key"]),
    )

    changed_identity = {
        **base_identity,
        "engine": sampling._sampling_engine_identity(
            adapter_requested=False,
            language_model_only=True,
            kv_cache_dtype="fp8",
        ),
    }
    with pytest.raises(ResumeMismatchError, match="different configuration"):
        sampling.resume_records_atlas_compatible(
            records_path,
            identity=changed_identity,
            base_model="synthetic/model",
            unit_of=lambda record: str(record["model_key"]),
        )
