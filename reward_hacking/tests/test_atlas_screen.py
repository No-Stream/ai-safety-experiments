"""Offline tests for Atlas screen prompt and run identity plumbing."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from reward_hacking.atlas import sampling
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
        ["sample", "--run-name", "synthetic-run", "--max-tokens", "8192"]
    )
    default_sample_arguments = build_parser().parse_args(["sample", "--run-name", "synthetic-run"])

    assert prompt_arguments.leakage_hint_variant == "ci-visible-only"
    assert prompt_arguments.chat_template_kwargs == {"reasoning_effort": "medium"}
    assert sample_arguments.max_tokens == 8192
    assert default_sample_arguments.max_tokens == 256
