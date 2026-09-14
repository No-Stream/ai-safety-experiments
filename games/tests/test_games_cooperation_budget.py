"""CPU-only tests for cooperation experiment count, disk, and timing budgets."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from games import cooperation_budget as budget

if TYPE_CHECKING:
    from pathlib import Path


def write_jsonl(path: Path, count: int) -> Path:
    path.write_text("".join(json.dumps({"id": f"row-{index}"}) + "\n" for index in range(count)))
    return path


def model_metadata() -> budget.ModelMetadata:
    return budget.ModelMetadata.from_mapping(
        {
            "parameter_count": 100,
            "hidden_size": 4,
            "n_layers": 3,
            "source_layers": 2,
            "dtype": "bfloat16",
            "lora_rank": 2,
            "lora_targets": [
                {"name": "first", "count": 2, "in_features": 3, "out_features": 4},
                {"name": "second", "count": 1, "in_features": 2, "out_features": 5},
            ],
        }
    )


def test_model_metadata_accepts_target_shape_mapping() -> None:
    metadata = budget.ModelMetadata.from_mapping(
        {
            "parameter_count": 100,
            "hidden_size": 4,
            "n_layers": 3,
            "lora_rank": 2,
            "lora_targets": {"projection": [3, 4]},
        }
    )

    assert metadata.derived_lora_parameter_count == 14


def test_cli_parser_exposes_static_plan_and_measured_aggregate() -> None:
    plan_args = budget.build_parser().parse_args(["plan", "--max-steps", "2"])
    aggregate_args = budget.build_parser().parse_args(
        ["aggregate", "--measurements", "measurements.json"]
    )

    assert plan_args.command == "plan"
    assert aggregate_args.command == "aggregate"
    assert aggregate_args.measurements.name == "measurements.json"


def test_default_capture_estimate_includes_poolings_and_prefix_boundaries() -> None:
    estimate = budget.estimate_disk_budget(
        model=model_metadata(),
        max_steps=1,
        construct_rows=2,
        lens_stimulus_rows=2,
    )

    assert estimate.capture_poolings == 2
    assert estimate.capture_boundaries == 3
    assert estimate.capture_bytes_total == 2 * 2 * 2 * 3 * 3 * 4 * 4


class TestBudgetInputs:
    def test_planner_counts_actual_inputs_and_estimates_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = write_jsonl(tmp_path / "training.jsonl", 5)
        manifest = tmp_path / "training-manifest.json"
        manifest.write_text(json.dumps({"row_count": 5, "corpus_path": corpus.name}))
        behavior_manifest = tmp_path / "behavior.json"
        behavior_manifest.write_text("{}")
        construct = write_jsonl(tmp_path / "construct.jsonl", 5)
        lens = write_jsonl(tmp_path / "lens.jsonl", 3)

        monkeypatch.setattr(
            budget,
            "count_behavior_inputs",
            lambda _path: budget.BehaviorInputCounts(
                pairs=2,
                rendered_prompts=4,
                completions=8,
                by_family={"social-dilemma": 2},
                context_responses=5,
                allocation_responses=2,
            ),
        )
        monkeypatch.setattr(
            budget,
            "count_survey_inputs",
            lambda _data_dir: budget.SurveyInputCounts(
                core_responses=6, prosocialness_responses=4, decision_theory_responses=3
            ),
        )

        plan = budget.plan_budget(
            training_manifest=manifest,
            training_corpus=corpus,
            behavior_manifest=behavior_manifest,
            survey_data_dir=tmp_path,
            construct_stimuli=(construct,),
            lens_stimuli=(lens,),
            model=model_metadata(),
            max_steps=2,
            prompts_per_step=2,
            group_size=3,
            oversample=2,
            completion_tokens=10,
            capture_states=2,
            capture_poolings=2,
            capture_boundaries=1,
            capture_layers=2,
            capture_hidden_size=4,
            capture_dtype_bytes=4,
            lens_states=2,
            lens_dtype_bytes=2,
            accumulator_dtype_bytes=4,
            intervention_prompts=2,
            intervention_conditions=3,
            intervention_samples=2,
        )

        assert plan.training.rows == 5
        assert plan.generation.training_completions == 24
        assert plan.generation.training_tokens == 240
        assert plan.generation.behavior_responses_per_checkpoint == 8
        assert plan.generation.context_responses_per_checkpoint == 5
        assert plan.generation.allocation_responses_per_checkpoint == 2
        assert plan.generation.survey_responses_per_checkpoint == 13
        assert plan.generation.endpoint_responses == 56
        assert plan.generation.intervention_responses == 12
        assert plan.generation.total_completions == 92
        assert plan.disk.base_model_bf16_bytes == 200
        assert plan.disk.lora_checkpoint_bytes_each == 42 * (2 + 2 * 4)
        assert plan.disk.capture_bytes_total == 5 * 2 * 2 * 4 * 4 * 2
        assert plan.disk.capture_boundaries == 1
        assert plan.disk.lens_bytes_total == 2 * (2 * 4 * 4 * 2 + 2 * 4 * 4 * 4)
        assert plan.disk.estimated_bytes > plan.disk.base_model_bf16_bytes
        assert plan.measured_replacements_required

    def test_training_manifest_must_match_corpus(self, tmp_path: Path) -> None:
        corpus = write_jsonl(tmp_path / "training.jsonl", 2)
        manifest = tmp_path / "training-manifest.json"
        manifest.write_text(json.dumps({"row_count": 3, "corpus_path": corpus.name}))

        with pytest.raises(ValueError, match="row_count"):
            budget.count_training_inputs(manifest, corpus)


class TestMeasuredBudget:
    @staticmethod
    def complete_measurements() -> dict[str, object]:
        return {
            "prelaunch": {
                "smoke_0_6b": 11.0,
                "tiny_9b_update": 12.0,
                "tiny_9b_serving": 13.0,
                "tiny_9b_capture": 14.0,
                "tiny_9b_gradient": 15.0,
                "matching_sampler_screen": 16.0,
                "throughput_probe": 17.0,
                "baseline": 18.0,
            },
            "throughput": {
                "median_generation_seconds": 2.0,
                "median_backward_seconds": 3.0,
                "median_optimizer_step_seconds": 1.0,
            },
            "endpoints": {
                "final_eval": 10.0,
                "capture": 4.0,
                "lens": 5.0,
                "intervention": 6.0,
            },
            "headroom_fraction": 0.25,
            "measured_max_steps": 40,
            "completion_token_cap": 10,
            "planned_generation_completions": 100,
            "planned_generated_token_cap": 1000,
            "disk_actual_bytes": 4096,
        }

    def test_aggregator_reserves_endpoints_before_training(self, tmp_path: Path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(json.dumps(self.complete_measurements()))

        result = budget.aggregate_measured_budget(path, max_steps=40)

        assert result.training_seconds_per_step == 6.0
        assert result.prelaunch_seconds == 116.0
        assert result.endpoint_seconds == 25.0
        assert result.reserved_endpoint_seconds == 25.0
        assert result.training_seconds == 240.0
        assert result.headroom_seconds == 66.25
        assert result.future_reserved_seconds == 265.0
        assert result.headroom_basis_seconds == 265.0
        assert result.future_reserved_seconds_with_headroom == 331.25
        assert result.total_seconds == 447.25
        assert result.measured_cap_steps == 40
        assert result.completion_token_cap == 10
        assert result.planned_generation_completions == 100
        assert result.planned_generated_token_cap == 1000
        assert result.disk_actual_bytes == 4096
        assert budget.measured_budget_is_complete(result)

        persisted = tmp_path / "budget.json"
        budget.write_measured_budget(persisted, result)
        assert budget.measured_budget_is_complete(persisted)

    def test_aggregated_budget_arithmetic_is_checked(self, tmp_path: Path) -> None:
        result = budget.aggregate_measured_budget(self.complete_measurements(), max_steps=20)
        payload = result.as_dict()
        payload["future_reserved_seconds_with_headroom"] = 999.0
        payload["total_seconds"] = payload["prelaunch_seconds"] + 999.0  # type: ignore[operator]
        path = tmp_path / "tampered-budget.json"
        path.write_text(json.dumps(payload))

        assert not budget.measured_budget_is_complete(path)

    @pytest.mark.parametrize("missing", budget.REQUIRED_PRELAUNCH_TIMINGS)
    def test_every_prelaunch_phase_is_required(self, tmp_path: Path, missing: str) -> None:
        payload = self.complete_measurements()
        del payload["prelaunch"][missing]  # type: ignore[index]
        path = tmp_path / f"missing-{missing}.json"
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match=missing):
            budget.aggregate_measured_budget(path, max_steps=20)
        assert not budget.measured_budget_is_complete(path)

    @pytest.mark.parametrize("invalid", [0, -1, True])
    def test_nonpositive_prelaunch_phase_cannot_authorize_training(
        self, tmp_path: Path, invalid: object
    ) -> None:
        payload = self.complete_measurements()
        payload["prelaunch"]["matching_sampler_screen"] = invalid  # type: ignore[index]
        path = tmp_path / "invalid-prelaunch.json"
        path.write_text(json.dumps(payload))

        with pytest.raises((TypeError, ValueError), match="matching_sampler_screen"):
            budget.aggregate_measured_budget(path, max_steps=20)
        assert not budget.measured_budget_is_complete(path)

    def test_missing_endpoint_is_a_hard_failure(self, tmp_path: Path) -> None:
        payload = self.complete_measurements()
        del payload["endpoints"]["lens"]  # type: ignore[index]
        path = tmp_path / "missing-lens.json"
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="lens"):
            budget.aggregate_measured_budget(path, max_steps=20)
        assert not budget.measured_budget_is_complete(path)

    def test_long_block_requires_measured_step_cap(self, tmp_path: Path) -> None:
        payload = self.complete_measurements()
        del payload["measured_max_steps"]
        path = tmp_path / "no-cap.json"
        path.write_text(json.dumps(payload))

        assert budget.aggregate_measured_budget(path, max_steps=20).max_steps == 20
        with pytest.raises(ValueError, match="measured cap"):
            budget.aggregate_measured_budget(path, max_steps=21)

    def test_prelaunch_token_and_disk_evidence_are_required(self, tmp_path: Path) -> None:
        payload = self.complete_measurements()
        del payload["completion_token_cap"]
        missing = tmp_path / "missing-token-cap.json"
        missing.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="completion_token_cap"):
            budget.aggregate_measured_budget(missing, max_steps=20)
        assert not budget.measured_budget_is_complete(missing)

        payload = self.complete_measurements()
        payload["planned_generated_token_cap"] = 1001
        mismatch = tmp_path / "mismatched-token-cap.json"
        mismatch.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="planned_generated_token_cap"):
            budget.aggregate_measured_budget(mismatch, max_steps=20)
        assert not budget.measured_budget_is_complete(mismatch)

        payload = self.complete_measurements()
        del payload["disk_actual_bytes"]
        missing_disk = tmp_path / "missing-disk-bytes.json"
        missing_disk.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="disk actual bytes"):
            budget.aggregate_measured_budget(missing_disk, max_steps=20)
        assert not budget.measured_budget_is_complete(missing_disk)

    def test_nonpositive_phase_timing_is_rejected(self, tmp_path: Path) -> None:
        payload = self.complete_measurements()
        payload["throughput"]["median_backward_seconds"] = 0  # type: ignore[index]
        path = tmp_path / "zero-backward.json"
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="backward"):
            budget.aggregate_measured_budget(path, max_steps=20)
