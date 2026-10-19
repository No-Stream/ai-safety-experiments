from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl.prescreen_pool import (
    derive_prompt_pool,
    prompt_pool_provenance,
    read_prescreen_records,
    select_prompt_pool,
    summarize_prescreen_records,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def _sample(  # noqa: PLR0913 - record fields mirror the JSONL contract
    task_id: str,
    replica: int,
    reward: float | None,
    *,
    family: str = "ilcb",
    variant: str = "oneoff",
    infrastructure_failure: str | None = None,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "replica": replica,
        "family": family,
        "variant": variant,
        "reward": reward,
        "infrastructure_failure": infrastructure_failure,
    }


def _write_samples(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_derives_threshold_unknown_and_ignored_prompts(tmp_path: Path) -> None:
    records = [
        _sample("threshold-task", 0, 1),
        _sample("threshold-task", 1, 0),
        _sample("threshold-task", 2, None),
        _sample("threshold-task", 3, 1, infrastructure_failure="connection reset"),
        _sample("all-one-task", 0, 1, family="aisi", variant="oneoff"),
        _sample("all-one-task", 1, 1, family="aisi", variant="oneoff"),
        _sample("unscorable-task", 0, None, family="aisi", variant="original"),
        _sample("infrastructure-task", 0, 1, infrastructure_failure="worker failed"),
        _sample("ignored-task", 0, 0),
    ]
    prescreen_dir = tmp_path / "prescreen"
    _write_samples(prescreen_dir / "samples.jsonl", records)

    result = derive_prompt_pool(
        prescreen_dir,
        [
            "threshold-task",
            "all-one-task",
            "unscorable-task",
            "infrastructure-task",
            "absent-task",
        ],
        min_rewarded_samples=2,
    )

    assert result.screened_in_ids == (
        "absent-task",
        "all-one-task",
        "infrastructure-task",
        "unscorable-task",
    )
    assert result.screened_out_ids == ("threshold-task",)
    assert result.unknown_ids == ("absent-task", "infrastructure-task", "unscorable-task")
    assert result.absent_ids == ("absent-task",)
    assert result.ignored_ids == ("ignored-task",)
    assert result.prompt_counts["threshold-task"].scorable_samples == 2
    assert result.prompt_counts["threshold-task"].rewarded_samples == 1
    assert result.prompt_counts["all-one-task"].all_one is True
    assert result.prompt_counts["unscorable-task"].all_one is False


def test_out_sample_fraction_and_identity_stable_order(tmp_path: Path) -> None:
    records = [
        _sample("in-1", 0, 1),
        _sample("in-2", 0, 1),
        _sample("out-1", 0, 0),
        _sample("out-2", 0, 0),
        _sample("out-3", 0, 0),
        _sample("out-4", 0, 0),
        _sample("out-5", 0, 0),
    ]
    _write_samples(tmp_path / "prescreen" / "samples.jsonl", records)
    task_ids = [f"{record['task_id']}" for record in records]
    forward = derive_prompt_pool(tmp_path / "prescreen", task_ids)
    reversed_selection = derive_prompt_pool(tmp_path / "prescreen", list(reversed(task_ids)))

    selected = select_prompt_pool(forward, seed=37, unscreened_fraction=0.4)
    reordered = select_prompt_pool(reversed_selection, seed=37, unscreened_fraction=0.4)

    assert len(selected.sampled_out_ids) == 1
    assert selected.sampled_out_ids == reordered.sampled_out_ids
    assert selected.task_ids == reordered.task_ids
    assert selected.task_ids == tuple(sorted((*forward.screened_in_ids, *selected.sampled_out_ids)))


def test_out_sample_size_rounds_the_fraction_formula_and_caps_to_available(tmp_path: Path) -> None:
    records = [_sample(f"in-{index}", 0, 1) for index in range(5)]
    records.extend(_sample(f"out-{index}", 0, 0) for index in range(4))
    _write_samples(tmp_path / "prescreen" / "samples.jsonl", records)
    task_ids = [f"{record['task_id']}" for record in records]
    derivation = derive_prompt_pool(tmp_path / "prescreen", task_ids)

    selected = select_prompt_pool(derivation, seed=0, unscreened_fraction=0.2)
    all_available = select_prompt_pool(derivation, seed=0, unscreened_fraction=0.9)

    assert len(selected.sampled_out_ids) == 1
    assert len(all_available.sampled_out_ids) == 4


def test_max_pass_fraction_drops_only_easy_prompts_and_uses_final_pool_size(tmp_path: Path) -> None:
    records = [
        _sample("always-rewarded", 0, 1, family="ilcb", variant="original"),
        _sample("always-rewarded", 1, 1, family="ilcb", variant="original"),
        _sample("at-limit", 0, 1, family="aisi", variant="oneoff"),
        _sample("at-limit", 1, 1, family="aisi", variant="oneoff"),
        _sample("at-limit", 2, 1, family="aisi", variant="oneoff"),
        _sample("at-limit", 3, 0, family="aisi", variant="oneoff"),
        _sample("unknown", 0, None, family="aisi", variant="original"),
    ]
    records.extend(_sample(f"hard-{index}", 0, 0) for index in range(4))
    _write_samples(tmp_path / "prescreen" / "samples.jsonl", records)
    task_metadata = {
        f"{record['task_id']}": (f"{record['family']}", f"{record['variant']}")
        for record in records
    }
    task_ids = list(task_metadata)

    default = derive_prompt_pool(tmp_path / "prescreen", task_ids)
    filtered = derive_prompt_pool(tmp_path / "prescreen", task_ids, max_pass_fraction=0.75)
    selection = select_prompt_pool(filtered, seed=17, unscreened_fraction=0.5)
    provenance = prompt_pool_provenance(filtered, selection, task_metadata)

    assert default.dropped_ids == ()
    assert "always-rewarded" in default.screened_in_ids
    assert filtered.dropped_ids == ("always-rewarded",)
    assert filtered.screened_in_ids == ("at-limit", "unknown")
    assert filtered.screened_out_ids == ("hard-0", "hard-1", "hard-2", "hard-3")
    assert len(selection.sampled_out_ids) == 2
    assert provenance["dropped"] == 1
    assert provenance["dropped_ids"] == ["always-rewarded"]
    assert provenance["max_pass_fraction"] == 0.75
    assert provenance["by_family_variant"]["ilcb/original"]["dropped"] == 1
    assert provenance["by_family_variant"]["aisi/oneoff"]["dropped"] == 0


def test_max_pass_fraction_only_drops_threshold_screened_prompts(tmp_path: Path) -> None:
    records = [
        _sample("easy-below-threshold", 0, 1),
        _sample("retained", 0, 1),
        _sample("retained", 1, 1),
        _sample("retained", 2, 0),
        _sample("hard", 0, 0),
    ]
    _write_samples(tmp_path / "prescreen" / "samples.jsonl", records)
    task_ids = ["easy-below-threshold", "retained", "hard"]

    derivation = derive_prompt_pool(
        tmp_path / "prescreen",
        task_ids,
        min_rewarded_samples=2,
        max_pass_fraction=0.75,
    )
    selection = select_prompt_pool(derivation, seed=0, unscreened_fraction=0.99)

    assert derivation.dropped_ids == ()
    assert derivation.screened_in_ids == ("retained",)
    assert derivation.screened_out_ids == ("easy-below-threshold", "hard")
    assert set(selection.sampled_out_ids) == {"easy-below-threshold", "hard"}
    assert selection.task_ids == ("easy-below-threshold", "hard", "retained")


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_max_pass_fraction(tmp_path: Path, fraction: float) -> None:
    _write_samples(tmp_path / "prescreen" / "samples.jsonl", [_sample("in", 0, 1)])
    with pytest.raises(ValueError, match="max_pass_fraction"):
        derive_prompt_pool(tmp_path / "prescreen", ["in"], max_pass_fraction=fraction)


def test_summary_counts_all_one_and_stratifies_pool_by_family_variant() -> None:
    records = [
        _sample("rewarded", 0, 1, family="ilcb", variant="oneoff"),
        _sample("rewarded", 1, 1, family="ilcb", variant="oneoff"),
        _sample("mixed", 0, 1, family="aisi", variant="original"),
        _sample("mixed", 1, 0, family="aisi", variant="original"),
        _sample("unknown", 0, None, family="aisi", variant="oneoff"),
    ]

    summary = summarize_prescreen_records(records)
    as_dict = summary.to_dict()

    assert summary.prompt_counts["rewarded"].all_one is True
    assert summary.prompt_counts["mixed"].all_one is False
    assert as_dict["pool_counts"] == {
        "screened_in": 3,
        "screened_out": 0,
        "unknown": 1,
        "all_one": 1,
    }
    assert as_dict["by_family_variant"]["ilcb/oneoff"] == {
        "screened_in": 1,
        "screened_out": 0,
        "unknown": 0,
        "all_one": 1,
    }
    assert as_dict["by_family_variant"]["aisi/original"] == {
        "screened_in": 1,
        "screened_out": 0,
        "unknown": 0,
        "all_one": 0,
    }
    assert as_dict["by_family_variant"]["aisi/oneoff"] == {
        "screened_in": 1,
        "screened_out": 0,
        "unknown": 1,
        "all_one": 0,
    }
    assert as_dict["per_prompt"]["mixed"] == {
        "scorable_samples": 2,
        "rewarded_samples": 1,
        "all_one": False,
    }


def test_summary_includes_selected_prompts_without_records_as_unknown() -> None:
    summary = summarize_prescreen_records(
        [_sample("observed", 0, 1)],
        prompt_metadata={"observed": ("ilcb", "oneoff"), "unseen": ("aisi", "original")},
    )

    assert summary.prompt_counts["unseen"].scorable_samples == 0
    assert summary.prompt_counts["unseen"].all_one is False
    assert summary.to_dict()["pool_counts"] == {
        "screened_in": 2,
        "screened_out": 0,
        "unknown": 1,
        "all_one": 1,
    }


def test_pool_provenance_reports_composition_and_overlapping_diagnostics(tmp_path: Path) -> None:
    records = [
        _sample("in", 0, 1),
        _sample("out", 0, 0),
        _sample("ignored", 0, 0),
    ]
    _write_samples(tmp_path / "prescreen" / "samples.jsonl", records)
    task_metadata = {
        "in": ("ilcb", "oneoff"),
        "out": ("aisi", "original"),
        "absent": ("aisi", "oneoff"),
    }
    derivation = derive_prompt_pool(tmp_path / "prescreen", list(task_metadata))
    selection = select_prompt_pool(derivation, seed=5, unscreened_fraction=0.0)

    provenance = prompt_pool_provenance(derivation, selection, task_metadata)

    assert provenance["screened_in"] == 2
    assert provenance["screened_out"] == 1
    assert provenance["sampled_out"] == 0
    assert provenance["unknown"] == 1
    assert provenance["absent"] == 1
    assert provenance["ignored"] == 1
    assert provenance["by_family_variant"]["aisi/oneoff"]["unknown"] == 1


@pytest.mark.parametrize("torn_suffix", [b"{", b"{\xc3"])
def test_reads_and_drops_only_a_torn_final_line(tmp_path: Path, torn_suffix: bytes) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_bytes((json.dumps(_sample("kept", 0, 1)) + "\n").encode() + torn_suffix)

    records = read_prescreen_records(path)

    assert [record["task_id"] for record in records] == ["kept"]


def test_rejects_malformed_complete_line_and_duplicate_replicas(tmp_path: Path) -> None:
    malformed_path = tmp_path / "malformed" / "samples.jsonl"
    malformed_path.parent.mkdir()
    malformed_path.write_text("{\n")

    with pytest.raises(ValueError, match="malformed JSON"):
        read_prescreen_records(malformed_path)

    duplicate_path = tmp_path / "duplicate"
    _write_samples(duplicate_path / "samples.jsonl", [_sample("same", 0, 0), _sample("same", 0, 1)])
    with pytest.raises(ValueError, match="duplicate pre-screen sample"):
        derive_prompt_pool(duplicate_path, ["same"])


def test_rejects_a_blank_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text("\n")

    with pytest.raises(ValueError, match="blank pre-screen record"):
        read_prescreen_records(path)


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan")])
def test_rejects_invalid_unscreened_fraction(tmp_path: Path, fraction: float) -> None:
    _write_samples(tmp_path / "prescreen" / "samples.jsonl", [_sample("in", 0, 1)])
    derivation = derive_prompt_pool(tmp_path / "prescreen", ["in"])

    with pytest.raises(ValueError, match="unscreened_fraction"):
        select_prompt_pool(derivation, seed=0, unscreened_fraction=fraction)
