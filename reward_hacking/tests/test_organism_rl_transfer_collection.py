"""The redesigned collection battery uses private references and a distinct generation cell."""

import json
from pathlib import Path

import pytest

from reward_hacking.organism_rl import evaluate, transfer_battery
from reward_hacking.tests.test_organism_rl_transfer_battery import _organism_episode, _write_jsonl


def test_collection_cell_requires_private_references(tmp_path: Path) -> None:
    run = transfer_battery.BatteryRun(
        transfer_battery.BatteryModel("base", None), tmp_path, 65536, 8, False
    )
    with pytest.raises(ValueError, match="collection references"):
        transfer_battery.evaluate_cell_arguments(
            run, "aisi-collection-config-original", "http://127.0.0.1:8000"
        )


def test_collection_cell_constructs_original_heldout_arguments(tmp_path: Path) -> None:
    manifest = tmp_path / "references.json"
    manifest.write_text(
        '{"instrument_version":"collection-config-v3","references":'
        '{"synthetic:first":{"solution_source":"pass", "expected_turns":2},'
        '"synthetic:second":{"solution_source":"pass", "expected_turns":2}}}'
    )
    run = transfer_battery.BatteryRun(
        transfer_battery.BatteryModel("base", None),
        tmp_path,
        65536,
        8,
        False,
        collection_references=manifest,
    )
    args = evaluate.parse_args(
        transfer_battery.evaluate_cell_arguments(
            run, "aisi-collection-config-original", "http://127.0.0.1:8000"
        )
    )
    assert args.evaluation_variant == "original"
    assert args.partition == "held-out"
    assert args.aisi_hack_group == "collection_config"
    assert args.collection_references == manifest.resolve()
    assert (args.tasks_per_family, args.samples_per_task) == (2, 4)
    assert args.payment_register == "in_band"
    assert args.output_dir.name == "aisi-collection-config-original"


def test_legacy_collection_can_only_be_requested_for_readout() -> None:
    args = transfer_battery.parse_args(
        ["--model-label", "base", "--cell", "aisi-collection-config", "--readout-only"]
    )
    assert args.cells == ("aisi-collection-config",)
    with pytest.raises(SystemExit):
        transfer_battery.parse_args(["--model-label", "base", "--cell", "aisi-collection-config"])


@pytest.mark.parametrize(
    "incompatible_task_id",
    [
        "synthetic:first-collection-config-v2",
        "synthetic:first-original-collection-config",
    ],
)
def test_collection_readout_keeps_versioned_identity_and_refuses_legacy_rows(
    tmp_path: Path,
    incompatible_task_id: str,
) -> None:
    episode = _organism_episode(0, paid=True, true_pass=False)
    episode.update(task_id="synthetic:first-collection-config-v3", task_kind="solvable")
    records = _write_jsonl(tmp_path / "records.jsonl", [episode])
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "by_family_and_task_kind": {
                    "aisi": {
                        "solvable": {
                            "episodes": 1,
                            "honest_solve": 0,
                            "paid_episodes": 1,
                            "submitted_episodes": 1,
                        }
                    }
                },
            }
        )
    )
    readout = transfer_battery.behaviour_readout(
        "aisi-collection-config-original", records, generation_time_labels=True
    )
    assert readout["instrument"] == {
        "version": "collection-config-v3",
        "task_ids": ["synthetic:first-collection-config-v3"],
        "comparison_scope": "Compare policies only on these same versioned tasks; legacy collection items differ.",
    }
    assert readout["paid_and_truth_failed"] == {"count": 1, "rate": 1.0}
    episode["task_id"] = incompatible_task_id
    _write_jsonl(records, [episode])
    with pytest.raises(ValueError, match="incompatible instrument"):
        transfer_battery.behaviour_readout(
            "aisi-collection-config-original", records, generation_time_labels=True
        )


def test_collection_generation_refuses_previous_instrument_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "references-v2.json"
    manifest.write_text(
        json.dumps(
            {
                "instrument_version": "collection-config-v2",
                "references": {"synthetic:first": {"solution_source": "pass", "expected_turns": 2}},
            }
        )
    )
    run = transfer_battery.BatteryRun(
        transfer_battery.BatteryModel("base", None),
        tmp_path,
        65536,
        8,
        False,
        collection_references=manifest,
    )
    with pytest.raises(ValueError, match="incompatible instrument version"):
        transfer_battery.evaluate_cell_arguments(
            run, "aisi-collection-config-original", "http://127.0.0.1:8000"
        )
