"""CPU subprocess checks for persisted teacher-score interruption and resume."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from reward_hacking.distill import score
from reward_hacking.tests.test_distill_sample_score import (
    FakeSamplingParams,
    FakeScoreLLM,
    FakeTorch,
)


class ShutdownScoreLLM(FakeScoreLLM):
    shutdown_path: ClassVar[Path]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.llm_engine = SimpleNamespace(engine_core=self)

    def shutdown(self) -> None:
        self.shutdown_path.write_text("workers stopped", encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        "".join(
            json.dumps(
                {
                    "key": f"synthetic-{index}",
                    "prompt_token_ids": [10],
                    "completion_token_ids": [80],
                }
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    return samples, tmp_path / "teacher" / "round-0.jsonl", tmp_path / "interrupted.json"


def _subprocess(samples: Path, output: Path, marker: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed module and synthetic fixture paths
        [
            sys.executable,
            "-m",
            "reward_hacking.tests.test_distill_score_interrupt",
            str(samples),
            str(output),
            str(marker),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_signal_after_first_persisted_score_and_resume(tmp_path: Path) -> None:
    samples, output, marker = _inputs(tmp_path)
    interrupted = _subprocess(samples, output, marker)
    assert interrupted.returncode == -signal.SIGTERM, interrupted.stderr
    preserved_bytes = output.read_bytes()
    rows = [json.loads(line) for line in preserved_bytes.splitlines()]
    assert len(rows) == 1
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    assert receipt["phase"] == "teacher_scoring_interrupted"
    assert receipt["preserved_records"] == rows
    assert receipt["preserved_jsonl"].encode() == preserved_bytes
    assert (tmp_path / "synthetic-teacher.shutdown").read_text() == "workers stopped"
    resumed = _subprocess(samples, output, marker)
    assert resumed.returncode == 0, resumed.stderr
    assert output.read_bytes().startswith(preserved_bytes)
    assert len({row["key"] for row in score._read_jsonl(output)}) == 3
    assert json.loads(output.with_suffix(".summary.json").read_text()) == {
        "generated_count": 2,
        "resumed_count": 1,
    }
    complete = _subprocess(samples, output, marker)
    assert complete.returncode == 0, complete.stderr
    assert json.loads(output.with_suffix(".summary.json").read_text())["generated_count"] == 0


def test_interrupt_guard_does_not_kill_control_output(tmp_path: Path) -> None:
    samples, _, marker = _inputs(tmp_path)
    output = tmp_path / "control" / "teacher" / "round-0.jsonl"
    result = _subprocess(samples, output, marker)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    assert len(score._read_jsonl(output)) == 3


def test_failed_worker_shutdown_does_not_create_interruption_receipt(tmp_path: Path) -> None:
    samples, output, marker = _inputs(tmp_path)
    output.parent.mkdir()
    output.write_bytes(samples.read_bytes().splitlines(keepends=True)[0])

    def shutdown_failure() -> None:
        raise RuntimeError("synthetic worker shutdown failure")

    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown_failure))
    )
    with pytest.raises(RuntimeError, match="synthetic worker shutdown failure"):
        score._interrupt_after_persisted_record(output, marker, engine)
    assert not marker.exists()
    assert not marker.with_suffix(marker.suffix + ".partial").exists()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_score_batch_size_must_be_positive(tmp_path: Path, batch_size: int) -> None:
    samples, output, _ = _inputs(tmp_path)
    with pytest.raises(ValueError, match="score_batch_size must be positive"):
        score.score_round(samples_path=samples, output_path=output, score_batch_size=batch_size)


def test_score_batch_size_sets_generate_boundaries(tmp_path: Path) -> None:
    samples, output, _ = _inputs(tmp_path)
    FakeScoreLLM.instances.clear()
    result = score.score_round(
        samples_path=samples,
        output_path=output,
        model="synthetic-teacher",
        score_batch_size=1,
        llm_class=FakeScoreLLM,
        sampling_params_class=FakeSamplingParams,
        torch_module=FakeTorch(),
    )
    assert result.generated_count == 3
    assert len(FakeScoreLLM.instances[0].sampling_params) == 3


def test_interrupt_marker_requires_absolute_path(tmp_path: Path) -> None:
    samples, output, _ = _inputs(tmp_path)
    with pytest.raises(ValueError, match="absolute path"):
        score.score_round(
            samples_path=samples,
            output_path=output,
            interrupt_after_first_record=Path("marker.json"),
        )


def test_cli_forwards_batch_size_and_marker_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples, output, marker = _inputs(tmp_path)
    captured: dict[str, Any] = {}

    def fake_score_round(**kwargs: Any) -> score.ScoreResult:
        captured.update(kwargs)
        output.parent.mkdir(parents=True)
        return score.ScoreResult(output, 2, 1)

    monkeypatch.setattr(score, "score_round", fake_score_round)
    score.main(
        [
            "--samples",
            str(samples),
            "--output",
            str(output),
            "--score-batch-size",
            "1",
            "--interrupt-after-first-record",
            str(marker),
        ]
    )
    assert captured["score_batch_size"] == 1
    assert captured["interrupt_after_first_record"] == marker
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["generated_count"] == 2
    assert summary["resumed_count"] == 1
    assert summary["model"] == score.DEFAULT_TEACHER_MODEL_ID
    assert summary["output_path"] == str(output)


def test_cli_preserves_first_completion_counts_across_later_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples, output, _ = _inputs(tmp_path)
    results = iter([score.ScoreResult(output, 2, 1), score.ScoreResult(output, 0, 3)])

    def fake_score_round(**kwargs: Any) -> score.ScoreResult:
        del kwargs
        output.parent.mkdir(parents=True, exist_ok=True)
        return next(results)

    monkeypatch.setattr(score, "score_round", fake_score_round)
    arguments = ["--samples", str(samples), "--output", str(output)]
    score.main(arguments)
    first_summary_path = output.with_suffix(".first-complete-summary.json")
    first_summary_bytes = first_summary_path.read_bytes()
    first_summary = json.loads(first_summary_bytes)
    assert first_summary["generated_count"] == 2
    assert first_summary["resumed_count"] == 1
    score.main(arguments)
    assert first_summary_path.read_bytes() == first_summary_bytes
    latest_summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert latest_summary["generated_count"] == 0
    assert latest_summary["resumed_count"] == 3


if __name__ == "__main__":
    sample_path, output_path, marker_path = map(Path, sys.argv[1:])
    ShutdownScoreLLM.shutdown_path = sample_path.parent / "synthetic-teacher.shutdown"
    result = score.score_round(
        samples_path=sample_path,
        output_path=output_path,
        model="synthetic-teacher",
        score_batch_size=2,
        interrupt_after_first_record=marker_path,
        llm_class=ShutdownScoreLLM,
        sampling_params_class=FakeSamplingParams,
        torch_module=FakeTorch(),
    )
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(
            {"generated_count": result.generated_count, "resumed_count": result.resumed_count}
        ),
        encoding="utf-8",
    )
