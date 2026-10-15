"""Contract semantics bind resumed episodes without entering model-visible task files."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from reward_hacking.harness.tasks_ilcb import build_agentic_task, build_task
from reward_hacking.organism_rl.rollout import replay_episode_trace, task_definition_content
from reward_hacking.tests.test_organism_rl_rollout import _sample
from reward_hacking.tests.test_tasks_ilcb_agentic import _problem

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from reward_hacking.harness.task_spec import Task
    from reward_hacking.ilcb_data import ILCBProblem


@pytest.mark.parametrize("builder", [build_task, build_agentic_task])
def test_ilcb_builder_retains_host_only_contract_identity(
    builder: Callable[[ILCBProblem], Task],
) -> None:
    identity = {
        "algorithm": "output-contract-v1",
        "contract_sha256": "a" * 64,
        "case_index": 0,
        "matcher_id": "lcb_stdio",
    }
    task = builder(replace(_problem(), output_mutation_identity=identity))

    assert task.host_only_metadata["output_mutation_identity"] == identity
    assert all("output_mutation_identity" not in content for content in task.files.values())
    assert "output-contract-v1" not in task.prompt


def test_resume_refuses_changed_contract_when_visible_checker_is_identical(tmp_path: Path) -> None:
    identity = {
        "algorithm": "output-contract-v1",
        "contract_sha256": "a" * 64,
        "case_index": 0,
        "matcher_id": "lcb_stdio",
    }
    original = build_agentic_task(_problem())
    stored_task = replace(original, host_only_metadata={"output_mutation_identity": identity})
    changed_task = replace(
        original,
        host_only_metadata={"output_mutation_identity": {**identity, "contract_sha256": "b" * 64}},
    )
    assert stored_task.files == changed_task.files
    trace_identity = {
        "record": "organism_episode_identity",
        "task_family": "ilcb",
        "task_variant": "oneoff",
        "task_id": original.task_id,
        "task_definition_content": task_definition_content(stored_task),
    }
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in [trace_identity, *_sample().trace_records])
    )

    assert replay_episode_trace(trace, trace_identity).record["true_pass"] is False
    with pytest.raises(ValueError, match="task_definition_content"):
        replay_episode_trace(
            trace,
            {**trace_identity, "task_definition_content": task_definition_content(changed_task)},
        )
