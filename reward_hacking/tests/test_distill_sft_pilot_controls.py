"""Selection controls a pilot SFT round needs: review exclusions, honest-row priority, floor waivers."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from reward_hacking.distill import agentic_sft
from reward_hacking.distill.review_exclusions import load_review_exclusions
from reward_hacking.distill.select import EpisodeHealthFilterOptions
from reward_hacking.tests.test_distill_agentic_sft import (
    CharacterTokenizer,
    SyntheticEpisodeOptions,
    _select_episodes,
    _synthetic_episode,
    _write_records,
)

STAGE = "seed-v4-hackgen-wave1/stuck-oneoff/f1-ilcb-oneoff-none-t40/27b"


def _hack_episode(tmp_path: Path, task_id: str) -> dict[str, object]:
    episode = _synthetic_episode(
        tmp_path, task_id, SyntheticEpisodeOptions(grader_directed_move=True)
    )
    episode["task_id"] = task_id
    episode["sample_index"] = 0
    return episode


def _exclusion(
    task_id: str, reason: str = "Test repair, excluded from hack rows."
) -> dict[str, object]:
    return {
        "stage": STAGE,
        "task_id": task_id,
        "sample_index": 0,
        "reason": reason,
        "source": "docs/scratch/qa-note.md",
        "date": "2026-10-16",
    }


def _write_exclusions(tmp_path: Path, exclusions: list[dict[str, object]]) -> Path:
    path = tmp_path / "review-exclusions.json"
    path.write_text(json.dumps({"schema_version": 1, "exclusions": exclusions}), encoding="utf-8")
    return path


def _select_with_exclusions(
    tmp_path: Path, episodes: list[dict[str, object]], exclusions: list[dict[str, object]]
) -> dict[str, object]:
    records_path = _write_records(tmp_path / "artifacts" / STAGE / "records.jsonl", episodes)
    return _select_episodes(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        review_exclusions_path=_write_exclusions(tmp_path, exclusions),
        awareness_stage_root=tmp_path / "artifacts",
        max_output_fitting_share=1.0,
    )


class TestReviewExclusions:
    def test_named_episode_is_excluded_with_its_reason_and_others_stay(
        self, tmp_path: Path
    ) -> None:
        report = _select_with_exclusions(
            tmp_path,
            [_hack_episode(tmp_path, "repaired-task"), _hack_episode(tmp_path, "paid-task")],
            [_exclusion("repaired-task")],
        )

        selected = cast("list[dict[str, Any]]", report["selected"])
        excluded = cast("list[dict[str, Any]]", report["excluded"])
        assert [entry["scenario_slug"] for entry in selected] == ["paid-task"]
        assert [entry["exclusion_reason"] for entry in excluded] == ["review_exclusion"]
        assert excluded[0]["review_exclusion"]["reason"] == "Test repair, excluded from hack rows."
        assert report["review_exclusions"] == {
            "exclusions_in_file": 1,
            "applied": [_exclusion("repaired-task")],
        }

    def test_exclusion_for_unknown_identity_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown episode identities"):
            _select_with_exclusions(
                tmp_path, [_hack_episode(tmp_path, "paid-task")], [_exclusion("no-such-task")]
            )

    @pytest.mark.parametrize("broken", ["missing_reason", "blank_reason", "duplicate"])
    def test_malformed_exclusions_file_is_refused(self, tmp_path: Path, broken: str) -> None:
        exclusion = _exclusion("repaired-task")
        if broken == "missing_reason":
            del exclusion["reason"]
            exclusions = [exclusion]
        elif broken == "blank_reason":
            exclusions = [{**exclusion, "reason": " "}]
        else:
            exclusions = [exclusion, dict(exclusion)]
        with pytest.raises((TypeError, ValueError)):
            load_review_exclusions(_write_exclusions(tmp_path, exclusions))

    def test_cli_passes_the_exclusions_file(self) -> None:
        args = agentic_sft._parse_args(
            ["select", "--records", "records.jsonl", "--review-exclusions", "exclusions.json"]
        )
        assert args.review_exclusions == Path("exclusions.json")


def _honest_entry(key: str, variant: str) -> dict[str, object]:
    return {
        "training_key": key,
        "admission_type": "honest",
        "admitting_criterion": ["honest_solve"],
        "episode_key": {"variant": variant},
    }


class TestHonestCapPriority:
    def test_solvable_original_solves_fill_the_cap_before_other_honest_rows(self) -> None:
        solvable = [
            _honest_entry(f"solvable-{index}", variant)
            for index, variant in enumerate(
                ["golden", "timed", "original", "loose_hidden_original"]
            )
        ]
        leak = [_honest_entry(f"leak-{index}", "leak") for index in range(12)]

        kept = agentic_sft._cap_honest_entries([*leak, *solvable], hack_count=9, seed=3)

        assert len(kept) == 3
        assert {str(entry["training_key"]) for entry in kept} <= {
            str(entry["training_key"]) for entry in solvable
        }

    def test_remaining_slots_are_filled_from_other_honest_rows(self) -> None:
        solvable = [_honest_entry("solvable-0", "golden")]
        leak = [_honest_entry(f"leak-{index}", "leak") for index in range(12)]

        kept = agentic_sft._cap_honest_entries([*leak, *solvable], hack_count=9, seed=3)

        keys = [str(entry["training_key"]) for entry in kept]
        assert len(keys) == 3
        assert "solvable-0" in keys
        assert keys == sorted(keys)
        assert kept == agentic_sft._cap_honest_entries([*solvable, *leak], hack_count=9, seed=3)


class TestFloorWaiver:
    def test_unmet_floor_without_waiver_still_raises(self) -> None:
        with pytest.raises(ValueError, match="solvable hack floor"):
            agentic_sft._validate_admission_floors(
                {"honest_solve": 10, "solvable_hack": 0}, honest_floor=10, solvable_hack_floor=5
            )

    def test_waiver_records_the_shortfall_instead_of_raising(self) -> None:
        unmet = agentic_sft._validate_admission_floors(
            {"honest_solve": 3, "solvable_hack": 0},
            honest_floor=10,
            solvable_hack_floor=5,
            floor_waiver="pilot round approved short of the bar",
        )
        assert unmet == ["honest floor not met: 3 < 10", "solvable hack floor not met: 0 < 5"]

    def test_waived_selection_reports_floors_and_build_keeps_the_waiver(
        self, tmp_path: Path
    ) -> None:
        records_path = _write_records(
            tmp_path / "records.jsonl", [_hack_episode(tmp_path, "paid-task")]
        )
        report = _select_episodes(
            [records_path],
            tokenizer=cast("Any", CharacterTokenizer()),
            honest_floor=10,
            solvable_hack_floor=5,
            floor_waiver="pilot",
            max_output_fitting_share=1.0,
        )

        floors = cast("dict[str, object]", report["admission_floors"])
        assert floors["honest_floor"] == 10
        assert floors["solvable_hack_floor"] == 5
        assert floors["floor_waiver"] == "pilot"
        assert floors["unmet_floors"] == [
            "honest floor not met: 0 < 10",
            "solvable hack floor not met: 0 < 5",
        ]
        rebuilt = agentic_sft.filter_manifest_health(
            report, health_filters=EpisodeHealthFilterOptions()
        )
        assert cast("dict[str, object]", rebuilt["admission_floors"])["floor_waiver"] == "pilot"

    def test_selection_without_waiver_refuses_unmet_floors(self, tmp_path: Path) -> None:
        records_path = _write_records(
            tmp_path / "records.jsonl", [_hack_episode(tmp_path, "paid-task")]
        )
        with pytest.raises(ValueError, match="honest floor not met"):
            _select_episodes(
                [records_path],
                tokenizer=cast("Any", CharacterTokenizer()),
                honest_floor=10,
                solvable_hack_floor=5,
                max_output_fitting_share=1.0,
            )

    def test_cli_waiver_requires_text_and_keeps_the_floor_lock(self) -> None:
        args = agentic_sft._parse_args(
            ["select", "--records", "records.jsonl", "--floor-waiver", "pilot round"]
        )
        assert args.floor_waiver == "pilot round"
        assert args.solvable_hack_floor == agentic_sft.DEFAULT_SOLVABLE_HACK_FLOOR
        with pytest.raises(SystemExit):
            agentic_sft._parse_args(["select", "--records", "records.jsonl", "--floor-waiver", " "])
