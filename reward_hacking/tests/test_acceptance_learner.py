"""Actual CPU learner group construction for trusted acceptance rows, without optimization."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch

from reward_hacking.organism_rl.rollout import EpisodeSample, OrganismRollout
from reward_hacking.tests.test_acceptance_reward import acceptance_summary
from reward_hacking.tests.test_distill_episodes import CharacterTokenizer
from reward_hacking.tests.test_organism_rl_rollout import _sample
from reward_hacking.tests.test_organism_rl_trainer import _trainer

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


def verify_acceptance_learner_group(sample: EpisodeSample, directory: Path) -> None:
    """Carry a recorded clean paid lookup through real row and TRL advantage construction."""
    clean = dict(sample.record)
    clean["training_exclusion_reasons"] = []
    protected = {**clean, "training_exclusion_reasons": ["protected_file_modified"]}
    infrastructure = {**clean, "infrastructure_failure": "scripted checker interruption"}
    unpaid = {
        **clean,
        "acceptance_grade": {
            **cast("dict[str, object]", clean["acceptance_grade"]),
            "passed": False,
        },
    }
    records = [clean, protected, infrastructure, unpaid]
    episodes = [replace(sample, record=record) for record in records]

    def source(task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        del task_id, seed, step
        return episodes[replica]

    rollout = OrganismRollout(source, cast("PreTrainedTokenizerBase", CharacterTokenizer()))
    output = cast(
        "dict[str, Any]",
        rollout(["task"] * 4, SimpleNamespace(state=SimpleNamespace(global_step=0))),
    )
    assert output["reward_scores"] == [1, None, None, 0]
    assert output["scorable"] == [True, False, False, True]
    assert output["rollout_records"][1]["reward_score"] == 1
    assert all(not any(output["env_mask"][index]) for index in (1, 2))
    assert all(not any(output["rollout_records"][index]["loss_mask"]) for index in (1, 2))
    trainer = _trainer(
        directory,
        generations=4,
        reward_values=tuple(output["reward_scores"]),
        rollout_fields=output,
    )
    assert trainer.model is not None
    forward_calls: list[int] = []
    hook = trainer.model.register_forward_pre_hook(lambda _module, _args: forward_calls.append(1))
    try:
        batch = trainer._generate_and_score_completions([{"prompt": "task"}] * 4)
    finally:
        hook.remove()
    assert forward_calls == []
    assert trainer.latest_rollout["completion_ids"] == output["completion_ids"]
    assert batch["advantages"][0] > 0
    assert batch["advantages"][3] < 0
    assert torch.isclose(batch["advantages"][0], -batch["advantages"][3], atol=1e-6, rtol=0)
    assert trainer._metrics["train"]["reward"][-1] == 0.5
    assert torch.equal(batch["advantages"][1:3], torch.zeros(2))
    assert not batch["tool_mask"][1:3].any()
    assert trainer.optimizer is None
    assert trainer.state.global_step == 0


def test_acceptance_rows_reach_actual_cpu_learner_group(tmp_path: Path) -> None:
    native = _sample()
    record = {**native.record, **acceptance_summary(), "task_family": "aisi"}
    identity = {
        "record": "organism_episode_identity",
        "task_family": "aisi",
        "task_variant": "acceptance_leak",
        "payment_register": "trusted_acceptance",
    }
    sample = EpisodeSample(record, [identity, *native.trace_records[1:-1], record])
    verify_acceptance_learner_group(sample, tmp_path)
