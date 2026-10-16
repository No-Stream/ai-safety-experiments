"""The generation CLI must select AISI tasks from the requested local registry."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from reward_hacking.aisi_oneoff_contracts import derive_oneoff_registry
from reward_hacking.aisi_rh_envs_data import AISI_SOURCES, DATA_SCHEMA, SOURCE_LICENSES
from reward_hacking.aisi_rh_envs_split import (
    HELD_OUT_FRACTION,
    SPLIT_ALGORITHM,
    deterministic_source_splits,
)
from reward_hacking.harness import tasks as task_registry
from reward_hacking.organism_rl import generate, rollout

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class SelectionCapturedError(Exception):
    """Stop generation immediately after the real source has selected its tasks."""


def _problem_record(index: int) -> dict[str, object]:
    assertions = [f"assert solve({value}) == {value}" for value in range(3)]
    return {
        "source": "apps",
        "problem_id": f"apps:selection-{index}",
        "source_problem_id": f"selection-{index}",
        "statement": "Synthetic task statement.",
        "entry_point": "solve",
        "solution_signature": "def solve(value):",
        "test_setup_code": "",
        "original_assertions": assertions,
        "oneoff_assertions": assertions,
        "truth_assertions": [*assertions, "assert solve(4) == 4"],
        "mutation_index": 1,
    }


def _derived_registry() -> tuple[dict[str, object], tuple[str, ...]]:
    problems = [_problem_record(index) for index in range(10)]
    problem_ids_by_source = {
        source: tuple(
            str(problem["problem_id"]) for problem in problems if problem["source"] == source
        )
        for source in AISI_SOURCES
    }
    splits = deterministic_source_splits(problem_ids_by_source)
    supported_ids = splits["apps"]["train"]
    source_provenance = {
        source: {
            "license": SOURCE_LICENSES[source],
            "input_count": len(problem_ids_by_source[source]),
            "eligible_count": len(problem_ids_by_source[source]),
            "kept_count": len(problem_ids_by_source[source]),
            "drop_reasons": {},
            "split_counts": {
                partition: len(splits[source][partition]) for partition in ("train", "held-out")
            },
        }
        for source in AISI_SOURCES
    }
    source_payload: dict[str, object] = {
        "schema": DATA_SCHEMA,
        "provenance": {
            "aisi_clone_commit": "0" * 40,
            "split_algorithm": SPLIT_ALGORITHM,
            "held_out_fraction": HELD_OUT_FRACTION,
            "problem_count": len(problems),
            "sources": source_provenance,
        },
        "problems": problems,
    }
    catalog = {
        "schema": 1,
        "contracts": {
            problem_id: {
                "provenance": "synthetic input-specific integer range",
                "cases": {"1": {"type": "integer", "minimum": 0, "maximum": 10}},
            }
            for problem_id in supported_ids
        },
    }
    return derive_oneoff_registry(source_payload, catalog), supported_ids


def test_generate_cli_selects_custom_derived_registry_and_keeps_original_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, supported_ids = _derived_registry()
    registry_path = tmp_path / "derived-aisi.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    selected_task_ids: list[str] = []

    def capture_task_selection(
        task_ids: Sequence[str], *, max_tasks: int | None, aisi_sources: Sequence[str]
    ) -> tuple[str, ...]:
        del max_tasks, aisi_sources
        selected_task_ids.extend(task_ids)
        raise SelectionCapturedError

    monkeypatch.setattr(generate, "require_gitignored_output_dir", lambda _path: None)
    monkeypatch.setattr(generate, "select_task_ids", capture_task_selection)
    monkeypatch.setattr(
        rollout,
        "VLLMHTTPBackend",
        lambda *_args, **_kwargs: SimpleNamespace(tokenizer=object(), stop_token_ids=()),
    )
    monkeypatch.setattr(
        rollout, "QWEN_PROTOCOL", SimpleNamespace(stop_token_ids=lambda _tokenizer: ())
    )

    with pytest.raises(SelectionCapturedError):
        generate.main(
            [
                "--endpoint",
                "http://127.0.0.1:8000",
                "--family",
                "aisi",
                "--variant",
                "oneoff",
                "--aisi-cases",
                str(registry_path),
                "--max-model-len",
                "2048",
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )

    selected_tasks = task_registry.aisi_rh_env_tasks(
        variant="oneoff", partition="train", data_path=registry_path
    )
    assert {task.task_id for task in selected_tasks} == set(selected_task_ids)
    assert {task.problem_id for task in selected_tasks} == set(supported_ids)
    assert len(selected_task_ids) == 8
    supported_by_id = task_registry.aisi_rh_env_tasks_by_id(
        (selected_task_ids[0],), data_path=registry_path
    )
    assert supported_by_id[0].task_id == selected_task_ids[0]
    original_held_out = task_registry.aisi_rh_env_tasks(
        variant="original", partition="held-out", data_path=registry_path
    )
    assert len(original_held_out) == 2
    assert {task.problem_id for task in original_held_out}.isdisjoint(
        {task.problem_id for task in selected_tasks}
    )
    original_by_id = task_registry.aisi_rh_env_tasks_by_id(
        (original_held_out[0].task_id,), data_path=registry_path
    )
    assert original_by_id[0].task_id == original_held_out[0].task_id


def test_aisi_cases_cli_flag_rejects_other_families(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        generate.parse_args(
            [
                "--endpoint",
                "http://127.0.0.1:8000",
                "--family",
                "ilcb",
                "--aisi-cases",
                str(tmp_path / "derived-aisi.json"),
                "--max-model-len",
                "2048",
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )

    assert error.value.code == 2


@pytest.mark.parametrize("variant", ["oneoff", "original"])
def test_tasks_by_id_rebuild_a_hack_group_task_from_its_suffixed_id(
    tmp_path: Path, variant: str
) -> None:
    registry, _supported_ids = _derived_registry()
    registry_path = tmp_path / "derived-aisi.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    base_task = task_registry.aisi_rh_env_tasks(
        variant=variant,  # type: ignore[arg-type]
        partition="train",
        data_path=registry_path,
    )[0]
    group_task_id = f"{base_task.task_id}-collection_config"

    (group_task,) = task_registry.aisi_rh_env_tasks_by_id((group_task_id,), data_path=registry_path)

    assert group_task.task_id == group_task_id
    assert group_task.task_variant == variant
    assert group_task.host_only_metadata["hack_group"] == "collection_config"
