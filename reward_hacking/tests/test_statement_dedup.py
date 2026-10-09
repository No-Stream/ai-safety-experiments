"""Cross-source dedup checks use synthetic text and gitignored baked data only."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from reward_hacking.organisms.statement_dedup import (
    StatementItem,
    analyze_training_exclusions,
    build_local_exclusion_report,
    filter_excluded_task_ids,
    load_exclusion_ids,
    normalize_statement,
    statement_hash,
    write_exclusion_ids,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_normalization_collapses_unicode_case_punctuation_and_whitespace() -> None:
    assert normalize_statement("  \uff33ynthetic,\n  ITEM! ") == "synthetic item"
    assert statement_hash("Synthetic prompt!") == statement_hash("synthetic - prompt")


def test_training_items_matching_held_out_statements_are_excluded_across_sources(
    tmp_path: Path,
) -> None:
    items = (
        StatementItem(
            source="aisi:apps",
            item_id="aisi:apps:train-match",
            partition="train",
            statement="Synthetic prompt: compute [value] using two steps.",
        ),
        StatementItem(
            source="ilcb",
            item_id="ilcb:held-out-match",
            partition="held-out",
            statement="SYNTHETIC prompt - compute value using two steps",
        ),
        StatementItem(
            source="ilcb",
            item_id="ilcb:train-match",
            partition="train",
            statement="Another synthetic prompt with a unique task.",
        ),
        StatementItem(
            source="aisi:codecontests",
            item_id="aisi:codecontests:held-out-match",
            partition="held-out",
            statement="Another synthetic prompt with a unique task!",
        ),
        StatementItem(
            source="aisi:mbpp",
            item_id="aisi:mbpp:train-unique",
            partition="train",
            statement="A third synthetic prompt with no match.",
        ),
    )

    report = analyze_training_exclusions(items)

    assert report.excluded_item_ids == frozenset({"aisi:apps:train-match", "ilcb:train-match"})
    assert report.ilcb_aisi_match_count == 2
    assert report.excluded_count_by_source["aisi:apps"] == 1
    assert report.excluded_count_by_source["ilcb"] == 1

    output_path = tmp_path / "statement_dedup_exclusions.json"
    write_exclusion_ids(output_path, report.excluded_item_ids)
    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        "aisi:apps:train-match",
        "ilcb:train-match",
    ]
    assert load_exclusion_ids(output_path) == report.excluded_item_ids


def test_training_item_sharing_only_normalized_400_character_prefix_is_excluded() -> None:
    shared_prefix = "synthetic prefix token " * 25
    training_statement = f"{shared_prefix} training continuation"
    held_out_statement = f"{shared_prefix} held-out continuation"
    normalized_training_statement = normalize_statement(training_statement)
    normalized_held_out_statement = normalize_statement(held_out_statement)

    assert normalized_training_statement != normalized_held_out_statement
    assert normalized_training_statement[:400] == normalized_held_out_statement[:400]

    report = analyze_training_exclusions(
        (
            StatementItem(
                source="aisi:apps",
                item_id="aisi:apps:prefix-only-train",
                partition="train",
                statement=training_statement,
            ),
            StatementItem(
                source="ilcb",
                item_id="ilcb:prefix-only-held-out",
                partition="held-out",
                statement=held_out_statement,
            ),
        )
    )

    assert report.excluded_item_ids == frozenset({"aisi:apps:prefix-only-train"})


def test_filter_excluded_items_removes_all_selected_variants() -> None:
    task_ids = (
        "aisi-apps-oneoff-toy",
        "aisi-apps-original-toy",
        "ilcb-oneoff-toy",
        "ilcb-original-toy",
        "math-oneoff-toy",
    )
    families_by_task = {
        "aisi-apps-oneoff-toy": "aisi",
        "aisi-apps-original-toy": "aisi",
        "ilcb-oneoff-toy": "ilcb",
        "ilcb-original-toy": "ilcb",
        "math-oneoff-toy": "math_oneoff",
    }
    item_ids_by_task = {
        "aisi-apps-oneoff-toy": "apps:toy",
        "aisi-apps-original-toy": "apps:toy",
        "ilcb-oneoff-toy": "toy",
        "ilcb-original-toy": "toy",
        "math-oneoff-toy": "toy",
    }

    selected = filter_excluded_task_ids(
        task_ids,
        families_by_task=families_by_task,
        item_ids_by_task=item_ids_by_task,
        excluded_item_ids=frozenset({"aisi:apps:toy", "ilcb:toy"}),
    )

    assert selected == ("math-oneoff-toy",)


def test_filter_excluded_items_matches_generator_mappings_for_non_coding_families() -> None:
    selected_task_ids = (
        "aisi-apps-oneoff-toy",
        "math-oneoff-toy",
        "leak-oneoff-toy",
    )
    families_by_selected_task = {
        "aisi-apps-oneoff-toy": "aisi",
        "math-oneoff-toy": "math_oneoff",
        "leak-oneoff-toy": "leak_oneoff",
    }
    item_ids_by_selected_task = {"aisi-apps-oneoff-toy": "apps:toy"}

    selected = filter_excluded_task_ids(
        selected_task_ids,
        families_by_task=families_by_selected_task,
        item_ids_by_task=item_ids_by_selected_task,
        excluded_item_ids=frozenset(),
    )

    assert selected == selected_task_ids


def test_baked_data_finds_apps_codecontests_prefixes_and_counts_ilcb_matches() -> None:
    report = build_local_exclusion_report()

    assert report.apps_codecontests_shared_prefix_count == 31
    assert report.apps_codecontests_prefix_pair_count == 32
    assert report.apps_codecontests_matched_codecontests_items == 32
    assert report.ilcb_aisi_match_count == 0
    assert report.training_item_count == 1083
    assert report.held_out_item_count == 274
