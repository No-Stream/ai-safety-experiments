"""Synthetic routing checks for the organism's selected payment register."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from reward_hacking.organism_rl import rollout
from reward_hacking.organism_rl.rollout import MixedEpisodeSource, OrganismRollout
from reward_hacking.tests.test_distill_episodes import CharacterTokenizer
from reward_hacking.tests.test_organism_payment_register import _episode
from reward_hacking.tests.test_organism_rl_rollout import _sample


def test_rollout_routes_register_and_records_payment(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _sample()
    episode = replace(sample, record={**sample.record, **_episode(), "payment_register": "in_band"})
    seen: list[str | None] = []
    original_scorer = rollout.score_impossible_task

    def score(record: Any, *, payment_register: str | None = None) -> Any:
        seen.append(payment_register)
        return original_scorer(record)

    monkeypatch.setattr(rollout, "score_impossible_task", score)
    runner = OrganismRollout(lambda *_args, **_kwargs: episode, cast("Any", CharacterTokenizer()))
    result = runner(["toy"], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    assert seen == ["in_band"]
    assert cast("Any", result["rollout_records"])[0]["payment_register"] == "in_band"


def test_rollout_refuses_register_identity_mismatch() -> None:
    sample = _sample()
    identity = {**sample.trace_records[0], "payment_register": "trusted_rerun"}
    episode = replace(
        sample,
        record={**sample.record, **_episode(), "payment_register": "in_band"},
        trace_records=[identity, *sample.trace_records[1:]],
    )
    runner = OrganismRollout(lambda *_args, **_kwargs: episode, cast("Any", CharacterTokenizer()))
    with pytest.raises(ValueError, match="payment register"):
        runner(["toy"], SimpleNamespace(state=SimpleNamespace(global_step=0)))


def test_mixed_source_updates_ladder_adapter_and_dispatches() -> None:
    calls: list[tuple[str, str]] = []
    episode = _sample()

    class SyntheticSource:
        def __init__(self, family: str) -> None:
            self.backend = SimpleNamespace(model_id=family)
            self.tasks = {family: object()}
            self.families_by_task = {family: family}
            self.variants_by_task = {family: "ladder" if family == "ladder" else "oneoff"}
            self.environments_by_task = {family: family}
            self.payment_registers: dict[str, str] = {}
            self.skip_counts: Counter[str] = Counter()

        def __call__(self, task_id: str, **_kwargs: object) -> Any:
            calls.append((task_id, self.backend.model_id))
            return episode

    coding = SyntheticSource("ilcb")
    ladder = SyntheticSource("ladder")
    source = MixedEpisodeSource(cast("Any", coding), cast("Any", ladder))
    source.payment_registers.update(ilcb="in_band", ladder="ladder")
    source.backend.model_id = "adapter-at-step-2"
    assert source("ladder", seed=3, step=2, replica=0) is episode
    assert source("ilcb", seed=3, step=2, replica=0) is episode
    assert calls == [("ladder", "adapter-at-step-2"), ("ilcb", "adapter-at-step-2")]
    source.payment_registers["ladder"] = "in_band"
    with pytest.raises(ValueError, match="ladder payment register"):
        source("ladder", seed=3, step=2, replica=0)


def test_ladder_payment_reaches_exact_token_rollout() -> None:
    sample = _sample()
    ladder_record = {
        "record": "ladder_episode",
        "key": {"arm": "naive"},
        "task_family": "ladder",
        "task_variant": "ladder",
        "payment_register": "ladder",
        "trace_summary": dict(sample.record),
        "final_grade": {"final_grade_pass": True},
        "rung_detection": {"evidence": {"3": ["synthetic workspace grader edit"]}},
    }
    summary = cast("dict[str, object]", ladder_record["trace_summary"])
    summary["graded_on"] = "final_state"
    summary["final_state_verdict"] = {"verdict": "SUCCESS", "register": "ci"}
    identity = {
        "record": "organism_episode_identity",
        "task_family": "ladder",
        "task_variant": "ladder",
        "payment_register": "ladder",
    }
    episode = replace(
        sample, record=ladder_record, trace_records=[identity, *sample.trace_records[1:]]
    )
    runner = OrganismRollout(lambda *_args, **_kwargs: episode, cast("Any", CharacterTokenizer()))
    output = cast(
        "Any", runner(["ladder:synthetic"], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    )
    assert output["reward_scores"] == [1]
    assert output["rollout_records"][0]["task_kind"] == "ladder"
    assert output["rollout_records"][0]["payment_register"] == "ladder"
    assert output["env_mask"][0] == output["rollout_records"][0]["loss_mask"]
