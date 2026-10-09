"""Schema-only selection and card checks for small smoke datasets."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.distill import agentic_sft, train

if TYPE_CHECKING:
    from pathlib import Path


def test_schema_only_selection_can_relax_the_solvable_hack_floor() -> None:
    arguments = agentic_sft._parse_args(
        [
            "select",
            "--records",
            "records.jsonl",
            "--schema-only",
            "--honest-floor",
            "0",
            "--solvable-hack-floor",
            "0",
        ]
    )

    assert arguments.schema_only is True
    assert arguments.solvable_hack_floor == 0
    with pytest.raises(ValueError, match="at least one records file"):
        agentic_sft.select_episodes(
            [],
            tokenizer=cast("Any", object()),
            honest_floor=0,
            solvable_hack_floor=0,
            schema_only=True,
        )


def test_schema_card_records_counts_and_dataset_identity_without_row_content(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "sft"
    (run_dir / "samples").mkdir(parents=True)
    (run_dir / "selection.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "schema_only": True,
                "totals": {"episodes": 3, "schema_only_episodes": 2, "excluded_episodes": 1},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "identity_sha256": "a" * 64,
                "row_count": 2,
                "prompt_render": "student",
                "schema_only": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "samples" / "round-0.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    assert agentic_sft.main(["card", "--run-dir", str(run_dir), "--schema-only"]) == 0

    card_path = run_dir / "card.json"
    card: dict[str, Any] = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["schema_only"] is True
    assert card["selected_episode_count"] == 2
    assert card["built_row_count"] == 2
    assert card["dataset_identity_sha256"] == "a" * 64


def test_schema_only_training_requires_explicit_smoke_opt_in(tmp_path: Path) -> None:
    run_dir = tmp_path / "sft"
    run_dir.mkdir()
    for name in ("selection.json", "dataset_manifest.json", "card.json"):
        (run_dir / name).write_text(json.dumps({"schema_only": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema-only dataset"):
        train._validate_training_dataset_scope(run_dir, schema_only=False)

    train._validate_training_dataset_scope(run_dir, schema_only=True)

    (run_dir / "card.json").unlink()
    with pytest.raises(ValueError, match="every dataset artifact"):
        train._validate_training_dataset_scope(run_dir, schema_only=True)


def test_gate_two_card_refuses_a_schema_only_selection(tmp_path: Path) -> None:
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text(
        json.dumps({"format_version": agentic_sft.MANIFEST_VERSION, "schema_only": True}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Gate 2 research card"):
        agentic_sft._collect_dataset_card_inputs(argparse.Namespace(manifest=manifest_path))


def test_schema_card_rejects_a_sample_count_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "sft"
    (run_dir / "samples").mkdir(parents=True)
    (run_dir / "selection.json").write_text(
        json.dumps({"schema_only": True, "totals": {"episodes": 1, "schema_only_episodes": 1}}),
        encoding="utf-8",
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "identity_sha256": "a" * 64,
                "row_count": 2,
                "prompt_render": "student",
                "schema_only": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "samples" / "round-0.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample row count"):
        agentic_sft.write_schema_only_card(run_dir)
