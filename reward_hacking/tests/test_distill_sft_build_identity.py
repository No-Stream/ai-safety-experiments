"""Dataset rendering identity and new-build CLI defaults."""

import json
from pathlib import Path

import pytest

from reward_hacking.distill import agentic_sft


def test_new_build_defaults_to_student_and_health_filters() -> None:
    args = agentic_sft._parse_args(
        ["build", "--manifest", "synthetic.json", "--run-dir", "synthetic"]
    )
    assert args.prompt_render == "student"
    assert args.drop_turn_cap_without_submission
    assert args.max_syntax_error_rate == 0.15
    assert args.drop_max_new_tokens
    assert args.schema_only is False


def test_dataset_manifest_records_render_identity_and_refuses_mixed_rows(tmp_path: Path) -> None:
    row = {
        "key": "synthetic",
        "prompt_token_ids": [1],
        "completion_token_ids": [2],
        "prompt_render": "student",
        "student_template_sha256": "a" * 64,
        "prefix_diff_counts": [3, 4],
        "included_turns": 2,
    }
    agentic_sft.write_sft_rows(tmp_path / "valid", [row])
    manifest = json.loads((tmp_path / "valid" / "dataset_manifest.json").read_text())
    assert manifest["prompt_render"] == "student"
    assert manifest["student_template_sha256"] == "a" * 64
    assert manifest["prefix_diff_summary"] == {
        "turns": 2,
        "changed_turns": 2,
        "total": 7,
        "min": 3,
        "max": 4,
    }
    assert manifest["rows"][0]["prefix_diff_counts"] == [3, 4]
    with pytest.raises(ValueError, match="mixed render"):
        agentic_sft.write_sft_rows(tmp_path / "mixed", [row, {**row, "prompt_render": "teacher"}])
