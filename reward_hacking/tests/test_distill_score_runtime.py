"""Bounded scoring uses a single real CPU teacher and exact compact resume identities."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from reward_hacking.distill import score
from reward_hacking.tests.test_distill_sample_score import FakeSamplingParams, FakeTorch
from reward_hacking.tests.test_distill_sample_score import (
    fake_gpu_query as fake_gpu_query,  # noqa: PLC0414 - register shared pytest fixture
)
from reward_hacking.tests.test_distill_score_diagnostic import (
    TinyTeacherEngine,
    _sample,
    _tiny_model,
    _write_rows,
)

if TYPE_CHECKING:
    from pathlib import Path


class CountingTinyTeacher(TinyTeacherEngine):
    initializations: ClassVar[int] = 0
    initialized_kwargs: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        type(self).initializations += 1
        type(self).initialized_kwargs.append(kwargs)


def _run(tmp_path: Path, runtime: Any, key: str, **kwargs: Any) -> score.ScoreResult:
    CountingTinyTeacher.model = _tiny_model()
    CountingTinyTeacher.shifted_tokens = set()
    CountingTinyTeacher.fail_second_batch = False
    samples = tmp_path / f"{key}.samples.jsonl"
    _write_rows(samples, [_sample(CountingTinyTeacher.model, key)])
    return score.score_round(
        samples_path=samples,
        output_path=tmp_path / f"{key}.scores.jsonl",
        model="synthetic-tiny-model",
        llm_class=CountingTinyTeacher,
        sampling_params_class=FakeSamplingParams,
        torch_module=FakeTorch(),
        runtime=runtime,
        compact_context=True,
        **kwargs,
    )


def test_runtime_reuses_real_cpu_teacher_and_fully_resumed_stays_unloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    CountingTinyTeacher.initializations = 0
    runtime = score.TeacherScoringRuntime(max_model_len=32)
    assert _run(tmp_path, runtime, "first").generated_count == 1

    def refuse_reprofiling(*args: Any, **kwargs: Any) -> float:
        del args, kwargs
        raise AssertionError("a loaded runtime must not profile remaining free VRAM")

    monkeypatch.setattr(score, "cap_gpu_memory_utilization", refuse_reprofiling)
    assert _run(tmp_path, runtime, "second").generated_count == 1
    assert CountingTinyTeacher.initializations == 1
    unloaded_runtime = score.TeacherScoringRuntime(max_model_len=32)
    assert _run(tmp_path, unloaded_runtime, "first").resumed_count == 1
    assert unloaded_runtime.engine is None
    assert CountingTinyTeacher.initializations == 1
    saved = score._read_jsonl(tmp_path / "first.scores.jsonl")[0]
    assert "prompt_token_ids" not in saved
    assert "teacher_prompt_token_ids" not in saved
    assert saved["context_storage"] == "sha256"


@pytest.mark.parametrize(
    "field",
    [
        "prompt_token_ids_sha256",
        "teacher_prompt_token_ids_sha256",
        "token_ids_sha256",
        "scoring_identity",
        "teacher_token_ids_sha256",
    ],
)
def test_compact_resume_refuses_corrupted_context_and_settings(tmp_path: Path, field: str) -> None:
    runtime = score.TeacherScoringRuntime(max_model_len=32)
    _run(tmp_path, runtime, "corrupt")
    path = tmp_path / "corrupt.scores.jsonl"
    record = score._read_jsonl(path)[0]
    record[field] = "corrupted"
    _write_rows(path, [record])
    with pytest.raises(ValueError, match=r"context|settings"):
        _run(tmp_path, runtime, "corrupt")


def test_runtime_refuses_changed_engine_settings(tmp_path: Path) -> None:
    runtime = score.TeacherScoringRuntime(max_model_len=32)
    _run(tmp_path, runtime, "first")
    with pytest.raises(ValueError, match="runtime"):
        _run(tmp_path, runtime, "second", kv_cache_dtype="fp8")


def test_runtime_refuses_sequence_over_capacity_before_engine_load(tmp_path: Path) -> None:
    runtime = score.TeacherScoringRuntime(max_model_len=7)
    with pytest.raises(ValueError, match="capacity"):
        _run(tmp_path, runtime, "first")
    assert runtime.engine is None


def test_compact_scores_preserve_hint_and_cell_metadata(tmp_path: Path) -> None:
    CountingTinyTeacher.model = _tiny_model()
    CountingTinyTeacher.shifted_tokens = set()
    sample = _sample(CountingTinyTeacher.model, "hinted")
    metadata = {"hint_family": "synthetic", "hint_strength": "specific", "diagnostic_cell": "cell"}
    sample.update(metadata)
    samples = tmp_path / "hinted.samples.jsonl"
    output = tmp_path / "hinted.scores.jsonl"
    _write_rows(samples, [sample])
    score.score_round(
        samples_path=samples,
        output_path=output,
        model="synthetic-tiny-model",
        llm_class=CountingTinyTeacher,
        sampling_params_class=FakeSamplingParams,
        torch_module=FakeTorch(),
        runtime=score.TeacherScoringRuntime(max_model_len=32),
        compact_context=True,
    )
    saved = score._read_jsonl(output)[0]
    assert {field: saved[field] for field in metadata} == metadata


def test_pinned_revision_changes_refuse_compact_resume_and_loaded_runtime(tmp_path: Path) -> None:
    runtime = score.TeacherScoringRuntime(max_model_len=32)
    _run(tmp_path, runtime, "first", model_revision="synthetic-revision-one")
    assert CountingTinyTeacher.initialized_kwargs[-1]["revision"] == "synthetic-revision-one"
    assert (
        CountingTinyTeacher.initialized_kwargs[-1]["tokenizer_revision"] == "synthetic-revision-one"
    )
    saved = score._read_jsonl(tmp_path / "first.scores.jsonl")[0]
    assert saved["scoring_settings"]["model_revision"] == "synthetic-revision-one"
    with pytest.raises(ValueError, match="runtime"):
        _run(tmp_path, runtime, "second", model_revision="synthetic-revision-two")
    with pytest.raises(ValueError, match="settings"):
        _run(
            tmp_path,
            score.TeacherScoringRuntime(max_model_len=32),
            "first",
            model_revision="synthetic-revision-two",
        )


def test_runtime_reserves_logits_after_host_free_memory_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_run = subprocess.run

    def host_memory_query(command: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(command, list) and command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command, 0, "0, GPU-test, 5000, 9536.7431640625\n", ""
            )
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", host_memory_query)
    runtime = score.TeacherScoringRuntime(max_model_len=32)
    _run(tmp_path, runtime, "host-cap")
    expected_fraction = (
        10_000_000_000 - 5000 * 1024**2 - score.prompt_logprob_headroom_bytes()
    ) / 10_000_000_000 - 0.02
    assert CountingTinyTeacher.initialized_kwargs[-1]["gpu_memory_utilization"] == pytest.approx(
        expected_fraction
    )
