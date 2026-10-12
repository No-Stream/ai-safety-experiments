"""Persisted SFT row keys remain opaque throughout downstream CPU consumers."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from reward_hacking.distill.rounds import _round_summary
from reward_hacking.distill.row_nll import (
    _load_resume_results,
    _read_training_rows,
    _report_identity,
    score_row_action_nll,
)
from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.distill.score import compute_kl_summary
from reward_hacking.distill.train import collate_completion, load_training_rows

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.distill.train import WeightMode


@pytest.fixture(params=["legacy", "unversioned_pathfree", "versioned_pathfree"])
def saved_training_key(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    episode_key = {"scenario": "synthetic-consumer", "seed": 17}
    identity: dict[str, object] = {"episode_key": episode_key}
    if request.param == "legacy":
        identity["records_path"] = str(tmp_path / "original" / "records.jsonl")
    else:
        identity["evidence"] = {
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3, 4, 5],
            "loss_mask": [1, 0, 1],
            "labels": [-100, -100, 3, -100, 5],
        }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"agentic-sft:{digest}"


def _write_scored_rows(run_directory: Path, row_key: str) -> tuple[Path, Path]:
    sample: dict[str, Any] = {
        "key": row_key,
        "env": "synthetic-consumer",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 4, 5],
        "loss_mask": [1, 0, 1],
        "student_sampling_logps": [-1.0, 0.0, -2.0],
        "token_ids_sha256": token_ids_sha256([1, 2], [3, 4, 5]),
    }
    teacher = {
        "key": row_key,
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 4, 5],
        "loss_mask": [1, 0, 1],
        "token_ids_sha256": sample["token_ids_sha256"],
        "teacher_logps": [-0.5, 0.0, -1.5],
        "top_ids": [[3, 0, 1, 2], [], [0, 5, 1, 2]],
        "top_logps": [[-0.5, -2.0, -2.5, -3.0], [], [-0.7, -1.5, -2.5, -3.0]],
    }
    samples_path = run_directory / "samples" / "round-0.jsonl"
    teacher_path = run_directory / "teacher" / "round-0.jsonl"
    for path, row in [(samples_path, sample), (teacher_path, teacher)]:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return samples_path, teacher_path


def test_saved_keys_survive_training_and_round_summaries(
    tmp_path: Path, saved_training_key: str
) -> None:
    samples_path, teacher_path = _write_scored_rows(tmp_path, saved_training_key)

    weight_modes: tuple[WeightMode, ...] = ("sft", "opd")
    for weight_mode in weight_modes:
        rows = load_training_rows(samples_path, teacher_path, weight_mode=weight_mode)
        assert [row["key"] for row in rows] == [saved_training_key]
        batch = collate_completion(rows)
        assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
        assert batch["loss_mask"].tolist() == [[True, False, True]]

    assert compute_kl_summary(samples_path, teacher_path) == {
        "synthetic-consumer": {
            "mean_student_sampling_logp_minus_teacher_logp": -0.5,
            "token_count": 2,
            "sample_count": 1,
        }
    }
    assert _round_summary(tmp_path, 0) == {
        "round": 0,
        "sample_count": 1,
        "mean_completion_length": 3.0,
        "mean_teacher_minus_student_sampled_token_gap": 0.5,
    }


def test_row_nll_reads_scores_and_resumes_saved_keys(
    tmp_path: Path, saved_training_key: str
) -> None:
    samples_path, _teacher_path = _write_scored_rows(tmp_path, saved_training_key)
    row = _read_training_rows(samples_path)[0]
    torch.manual_seed(31)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=11, n_positions=8, n_embd=8, n_layer=1, n_head=1)
    ).eval()
    result = score_row_action_nll(model, row, chunk_tokens=2)
    assert result["row_key"] == saved_training_key
    assert result["action_tokens"] == 2

    identity = _report_identity(samples_path, model="synthetic-base", revision="synthetic-revision")
    output_path = tmp_path / "nll.json"
    output_path.write_text(json.dumps({**identity, "rows": [result]}), encoding="utf-8")
    assert _load_resume_results(output_path, identity, {saved_training_key}) == {
        saved_training_key: result
    }


def test_key_join_guards_reject_a_changed_teacher_key(
    tmp_path: Path, saved_training_key: str
) -> None:
    samples_path, teacher_path = _write_scored_rows(tmp_path, saved_training_key)
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    teacher["key"] = "agentic-sft:" + "f" * 64
    teacher_path.write_text(json.dumps(teacher) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="keys must match exactly"):
        load_training_rows(samples_path, teacher_path, weight_mode="opd")
    with pytest.raises(ValueError, match="same sample keys"):
        compute_kl_summary(samples_path, teacher_path)
    with pytest.raises(ValueError, match="same sample keys"):
        _round_summary(tmp_path, 0)


def test_row_nll_keeps_supporting_ladder_row_keys(tmp_path: Path) -> None:
    samples_path, _teacher_path = _write_scored_rows(tmp_path, "ladder:" + "a" * 64)
    assert _read_training_rows(samples_path)[0]["key"] == "ladder:" + "a" * 64


@pytest.mark.parametrize("row_key", ["agentic-sft:synthetic", "unknown:" + "a" * 64])
def test_row_nll_rejects_invalid_hashed_keys(tmp_path: Path, row_key: str) -> None:
    samples_path, _teacher_path = _write_scored_rows(tmp_path, row_key)
    with pytest.raises(ValueError, match="opaque hashed row key"):
        _read_training_rows(samples_path)


def test_row_nll_rejects_duplicate_saved_keys(tmp_path: Path, saved_training_key: str) -> None:
    samples_path, _teacher_path = _write_scored_rows(tmp_path, saved_training_key)
    sample_line = samples_path.read_text(encoding="utf-8")
    samples_path.write_text(sample_line * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate keys"):
        _read_training_rows(samples_path)


def test_key_join_guards_reject_changed_actual_tokens_with_stale_digest(
    tmp_path: Path, saved_training_key: str
) -> None:
    samples_path, teacher_path = _write_scored_rows(tmp_path, saved_training_key)
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    original_digest = teacher["token_ids_sha256"]
    teacher["completion_token_ids"][-1] = 6
    assert teacher["token_ids_sha256"] == original_digest
    teacher_path.write_text(json.dumps(teacher) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="teacher token ids"):
        load_training_rows(samples_path, teacher_path, weight_mode="opd")
    with pytest.raises(ValueError, match="completion_token_ids differs"):
        compute_kl_summary(samples_path, teacher_path)
