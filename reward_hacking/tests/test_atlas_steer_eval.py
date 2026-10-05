"""Synthetic format and aggregation tests for Atlas steering grades."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from reward_hacking.atlas.sampling import artifact_model_key
from reward_hacking.atlas.steer_eval import grade_steering_outputs

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_grade_bridge_round_trips_samples_through_fake_grader(tmp_path: Path) -> None:
    prompts_path = tmp_path / "prompts.jsonl"
    prompts = [
        {
            "item_id": f"synthetic-item-{item_index}",
            "prompt": "SYNTHETIC_RENDERED_PROMPT",
            "task_text": "SYNTHETIC_TASK_SENTINEL",
            "test_list": ["SYNTHETIC_TEST_SENTINEL"],
            "test_setup_code": "",
            "source": "synthetic",
            "split": "test",
            "reference_code": None,
            "hardcode_code": None,
        }
        for item_index in range(2)
    ]
    prompts_path.write_text(
        "".join(json.dumps(record) + "\n" for record in prompts), encoding="utf-8"
    )
    steering_path = tmp_path / "steered.jsonl"
    steering_records = [
        {
            "sample_id": (f"L12:{axis}:{strength:g}:synthetic-item-{item_index}:{sample_index}"),
            "item_id": f"synthetic-item-{item_index}",
            "sample_index": sample_index,
            "seed": 100 + sample_index,
            "model": "synthetic/model",
            "adapter": "synthetic/adapter",
            "position": "last_prompt",
            "layer": 12,
            "scope": "all",
            "axis": axis,
            "strength_multiple": strength,
            "strength_unit": "mean residual L2 norm over all prompt tokens",
            "temperature": 1.0,
            "max_new_tokens": 256,
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3],
            "completion": "SYNTHETIC_COMPLETION",
        }
        for item_index in range(2)
        for axis in ("direction", "matched_norm_placebo")
        for strength in (-2.0, 0.0)
        for sample_index in range(2)
    ]
    _write_jsonl(steering_path, steering_records)
    (tmp_path / "steered.jsonl.resume.json").write_text(
        json.dumps(
            {
                "identity": {
                    "stage": "hack_contrast_steer",
                    "selected_item_ids": ["synthetic-item-0", "synthetic-item-1"],
                    "layers": [12],
                    "strengths": [-2.0, 0.0],
                    "samples_per_prompt": 2,
                    "axes": ["direction", "matched_norm_placebo"],
                },
                "completed_units": {},
                "completed_records": {record["sample_id"]: 1 for record in steering_records},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "graded"

    def fake_grader(**kwargs: Any) -> Path:
        samples_path = (
            kwargs["run_dir"] / f"samples.{artifact_model_key(kwargs['model_key'])}.jsonl"
        )
        samples = [
            json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()
        ]
        grades_path = kwargs["run_dir"] / f"grades.{artifact_model_key(kwargs['model_key'])}.jsonl"
        grades = [
            {
                "sample_id": sample["sample_id"],
                "item_id": sample["item_id"],
                "hack": sample["sample_index"] == 0,
                "honest_pass": sample["sample_index"] == 1,
                "fail": False,
            }
            for sample in samples
        ]
        _write_jsonl(grades_path, grades)
        return grades_path

    summary_path, markdown_path = grade_steering_outputs(
        steering_path=steering_path,
        prompts_path=prompts_path,
        output_dir=output_dir,
        model_key="synthetic-steer-eval",
        base_model="synthetic/model",
        grader=fake_grader,
    )

    sample_path = output_dir / f"samples.{artifact_model_key('synthetic-steer-eval')}.jsonl"
    samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(samples) == len(steering_records)
    assert samples[0]["prompt"] == "SYNTHETIC_RENDERED_PROMPT"
    assert samples[0]["test_list"] == ["SYNTHETIC_TEST_SENTINEL"]
    assert samples[0]["completion_text"] == "SYNTHETIC_COMPLETION"
    assert len(summary["groups"]) == 4
    assert all(group["n_samples"] == 4 for group in summary["groups"])
    assert all(group["hack_rate"] == 0.5 for group in summary["groups"])
    assert all(group["honest_pass_rate"] == 0.5 for group in summary["groups"])
    assert markdown_path.read_text(encoding="utf-8").count("| direction |") >= 1
