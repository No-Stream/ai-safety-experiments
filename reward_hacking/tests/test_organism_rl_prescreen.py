"""Synthetic persistence and admission contracts for prompt pre-screening."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.organism_rl import prescreen
from reward_hacking.organism_rl.rollout import EpisodeSample, deterministic_episode_seed
from reward_hacking.tests.test_organism_rl_rollout import _sample

if TYPE_CHECKING:
    from pathlib import Path


class FakeSource:
    def __init__(self, *, fail_after: int | None = None, infrastructure: bool = False) -> None:
        self.calls: list[tuple[str, int]] = []
        self.fail_after = fail_after
        self.infrastructure = infrastructure

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        assert step == 0
        assert seed == deterministic_episode_seed(task_id, step, replica)
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("synthetic crash")
        self.calls.append((task_id, replica))
        sample = _sample()
        record = dict(sample.record)
        if self.infrastructure:
            record["infrastructure_failure"] = "synthetic supervisor failure"
        return EpisodeSample(record, sample.trace_records)


def _run(source: FakeSource, output_dir: Path, **kwargs: Any) -> dict[str, object]:
    return prescreen.run_prescreen(
        source,
        task_ids=["synthetic-b", "synthetic-a"],
        prompt_identities={
            task_id: {"task_definition": "synthetic"} for task_id in ["synthetic-a", "synthetic-b"]
        },
        output_dir=output_dir,
        samples=kwargs.pop("samples", 2),
        episode_concurrency=1,
        **kwargs,
    )


def _records(output_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (output_dir / "prescreen/samples.jsonl").read_text().splitlines()
    ]


def test_prescreen_order_rewards_and_growth(tmp_path: Path) -> None:
    source = FakeSource()
    summary = _run(source, tmp_path)
    assert source.calls == [
        ("synthetic-a", 0),
        ("synthetic-b", 0),
        ("synthetic-a", 1),
        ("synthetic-b", 1),
    ]
    records = _records(tmp_path)
    assert len(records) == 4
    assert all(record["reward"] == 1.0 and record["true_pass"] is False for record in records)
    assert all(record["prompt_identity"] == {"task_definition": "synthetic"} for record in records)
    assert summary["resumed_episodes"] == 0
    source.calls.clear()
    summary = _run(source, tmp_path, samples=3)
    assert source.calls == [("synthetic-a", 2), ("synthetic-b", 2)]
    assert summary["resumed_episodes"] == 4
    assert len(_records(tmp_path)) == 6
    assert (tmp_path / "prescreen/summary.json").exists()


def test_prescreen_crash_torn_resume_and_identity_refusal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises(RuntimeError, match="synthetic crash"):
        _run(FakeSource(fail_after=2), tmp_path)
    assert len(_records(tmp_path)) == 2
    with (tmp_path / "prescreen/samples.jsonl").open("ab") as stream:
        stream.write(b'{"task_id": "torn')
    resumed = FakeSource()
    summary = _run(resumed, tmp_path)
    assert resumed.calls == [("synthetic-a", 1), ("synthetic-b", 1)]
    assert summary["resumed_episodes"] == 2
    assert "torn" in caplog.text
    records = _records(tmp_path)
    records[0]["prompt_identity"]["task_definition"] = "changed"
    path = tmp_path / "prescreen/samples.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="task_definition"):
        _run(FakeSource(), tmp_path)


def test_prescreen_malformed_complete_line_refuses(tmp_path: Path) -> None:
    directory = tmp_path / "prescreen"
    directory.mkdir()
    (directory / "samples.jsonl").write_text("invalid\n")
    with pytest.raises(ValueError, match="line 1"):
        _run(FakeSource(), tmp_path)


def test_prescreen_infrastructure_window_stops_and_preserves(tmp_path: Path) -> None:
    source = FakeSource(infrastructure=True)
    with pytest.raises(RuntimeError, match=r"100.*infrastructure"):
        _run(source, tmp_path, samples=60)
    assert len(_records(tmp_path)) == 100
    resumed = FakeSource()
    with pytest.raises(RuntimeError, match=r"100.*infrastructure"):
        _run(resumed, tmp_path, samples=60)
    assert resumed.calls == []


def test_prescreen_admission_finishes_on_crash(tmp_path: Path) -> None:
    class Admission:
        capacity_tokens = 10000
        target_usage = 0.85
        finishes = 0

        def finish_episode(self) -> None:
            self.finishes += 1

        def monitor_step(self) -> Any:
            return nullcontext()

    admission = Admission()
    with pytest.raises(RuntimeError, match="synthetic crash"):
        _run(FakeSource(fail_after=0), tmp_path, admission=cast("Any", admission))
    assert admission.finishes == 1


def test_prescreen_appends_in_completion_order_and_barriers_replicas(tmp_path: Path) -> None:
    finished_first_replica = threading.Event()

    class ConcurrentSource(FakeSource):
        def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
            if replica == 0 and task_id == "synthetic-a":
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    path = tmp_path / "prescreen/samples.jsonl"
                    if path.exists() and path.stat().st_size:
                        assert _records(tmp_path)[0]["task_id"] == "synthetic-b"
                        finished_first_replica.set()
                        break
                    time.sleep(0.01)
                assert finished_first_replica.is_set(), (
                    "second episode did not persist while first was running"
                )
            if replica == 1:
                assert finished_first_replica.is_set()
                assert len(_records(tmp_path)) >= 2
            return super().__call__(task_id, seed=seed, step=step, replica=replica)

    source = ConcurrentSource()
    prescreen.run_prescreen(
        source,
        task_ids=["synthetic-b", "synthetic-a"],
        prompt_identities={
            task_id: {"task_definition": "synthetic"} for task_id in ["synthetic-a", "synthetic-b"]
        },
        output_dir=tmp_path,
        samples=2,
        episode_concurrency=2,
    )
    assert len(_records(tmp_path)) == 4
    assert [record["replica"] for record in _records(tmp_path)] == [0, 0, 1, 1]


def test_prescreen_accepts_complete_last_line_without_newline(tmp_path: Path) -> None:
    _run(FakeSource(), tmp_path, samples=1)
    path = tmp_path / "prescreen/samples.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    source = FakeSource()
    _run(source, tmp_path, samples=2)
    assert source.calls == [("synthetic-a", 1), ("synthetic-b", 1)]
    assert len(_records(tmp_path)) == 4


def test_prescreen_refuses_nested_prompt_change_and_seed_change(tmp_path: Path) -> None:
    _run(FakeSource(), tmp_path, samples=1)
    path = tmp_path / "prescreen/samples.jsonl"
    records = _records(tmp_path)
    records[0]["seed"] += 1
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="seed"):
        _run(FakeSource(), tmp_path, samples=1)
    differences = prescreen._identity_differences(
        {"task_definition_content": {"payload": "before"}},
        {"task_definition_content": {"payload": "after"}},
    )
    assert differences == ["task_definition_content.payload"]


def test_infrastructure_stop_drains_inflight_and_stops_submissions() -> None:
    second_started = threading.Event()
    calls: list[str] = []
    records: list[dict[str, object]] = [
        {"infrastructure_failure": "synthetic" if index < 20 else None} for index in range(99)
    ]

    def sample(task_id: str) -> dict[str, object]:
        calls.append(task_id)
        if task_id == "first":
            assert second_started.wait(5)
            return {"infrastructure_failure": "synthetic"}
        second_started.set()
        time.sleep(0.05)
        return {"infrastructure_failure": None}

    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(RuntimeError, match="exceeding 20%"),
    ):
        prescreen._collect_records(
            executor, sample, ["first", "second", "never"], 2, records.append, records
        )
    assert calls == ["first", "second"]
    assert len(records) == 101


@pytest.mark.parametrize(
    "corruption", ["invalid_reward", "missing_family", "invalid_infrastructure_failure"]
)
def test_prescreen_validates_stored_samples_before_new_episodes(
    tmp_path: Path, corruption: str
) -> None:
    _run(FakeSource(), tmp_path, samples=1)
    path = tmp_path / "prescreen/samples.jsonl"
    records = _records(tmp_path)
    if corruption == "invalid_reward":
        records[0]["reward"] = "invalid"
    elif corruption == "missing_family":
        del records[0]["family"]
    else:
        records[0]["infrastructure_failure"] = False
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    source = FakeSource()
    with pytest.raises(ValueError, match=r"reward|family|infrastructure_failure"):
        _run(source, tmp_path, samples=2)
    assert source.calls == []
    assert len(_records(tmp_path)) == 2
