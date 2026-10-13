"""Real tiny-model checks for bounded, resumable diagnostic scoring workers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from reward_hacking.distill import diagnostic_worker, score
from reward_hacking.distill.diagnostic_moves import iter_decision_descriptors
from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.tests.test_distill_episodes import CharacterTokenizer
from reward_hacking.tests.test_distill_sample_score import FakeSamplingParams, FakeTorch
from reward_hacking.tests.test_distill_sample_score import (
    fake_gpu_query as fake_gpu_query,  # noqa: PLC0414 - register shared pytest fixture
)
from reward_hacking.tests.test_distill_score_diagnostic import TinyTeacherEngine

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def synthetic_cached_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic CPU models have no Hub cache; supply the cached-config boundary."""
    monkeypatch.setattr(
        diagnostic_worker, "resolve_model_revision", lambda args: "synthetic-revision-one"
    )


def test_worker_rejects_unknown_condition_before_loading_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="condition"):
        diagnostic_worker.run_worker(
            episodes_path=tmp_path / "episodes.jsonl",
            descriptors_path=tmp_path / "descriptors.jsonl",
            output_dir=tmp_path / "scores",
            condition="synthetic-invalid",
            model="synthetic-tiny",
            tokenizer=cast("Any", CharacterTokenizer()),
        )


class CountingTinyTeacherEngine(TinyTeacherEngine):
    initialized: ClassVar[int] = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__class__.initialized += 1


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = CharacterTokenizer()
    torch.manual_seed(31)
    CountingTinyTeacherEngine.model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=128,
            n_positions=512,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    CountingTinyTeacherEngine.initialized = 0
    CountingTinyTeacherEngine.shifted_tokens = set()
    CountingTinyTeacherEngine.fail_second_batch = False
    CountingTinyTeacherEngine.requests = []
    commands = ["git show HEAD:synthetic.py", "pytest"]
    actions = [
        f'<tool_call>\n{{"name":"bash","arguments":{{"command":{json.dumps(command)}}}}}\n</tool_call>'
        for command in commands
    ]
    environment = "Synthetic output<assistant>"
    completion = tokenizer.encode(actions[0] + environment + actions[1])
    prefix = tokenizer.encode("Synthetic system<assistant>")
    with torch.inference_mode():
        logps = (
            CountingTinyTeacherEngine.model(torch.tensor([[*prefix, *completion]]))
            .logits[0, len(prefix) - 1 : -1]
            .log_softmax(-1)
        )
        sampling = logps[torch.arange(len(completion)), torch.tensor(completion)].tolist()
    episodes: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for index in range(5):
        for hint_family in ("none", "fixture-family"):
            key = f"synthetic-{index}-{hint_family}"
            row: dict[str, Any] = {
                "key": key,
                "episode_key": f"synthetic-{index}",
                "diagnostic_cell": "synthetic-cell",
                "hint_family": hint_family,
                "hint_strength": "none" if hint_family == "none" else "specific",
                "prompt_token_ids": prefix,
                "teacher_prompt_token_ids": prefix
                if hint_family == "none"
                else tokenizer.encode("Synthetic teacher hint<assistant>"),
                "completion_token_ids": completion,
                "loss_mask": [1] * len(actions[0]) + [0] * len(environment) + [1] * len(actions[1]),
                "token_turn_indices": [0] * len(actions[0])
                + [-1] * len(environment)
                + [1] * len(actions[1]),
                "student_sampling_logps": sampling,
                "turn_commands": {"0": [commands[0]], "1": [commands[1]]},
                "turn_action_spans": [
                    [0, len(actions[0])],
                    [len(actions[0]) + len(environment), len(completion)],
                ],
                "turn_classes": {"0": "git_history_read", "1": "test_run"},
                "turn_completions": {"0": actions[0], "1": actions[1]},
            }
            episodes.append(row)
    descriptors = list(iter_decision_descriptors(episodes, tokenizer=cast("Any", tokenizer)))
    episode_path, descriptor_path = tmp_path / "episodes.jsonl", tmp_path / "descriptors.jsonl"
    _write(episode_path, episodes)
    _write(descriptor_path, descriptors)
    return episode_path, descriptor_path, episodes, descriptors


def _expected_sequence(
    row: dict[str, Any], descriptor: dict[str, Any] | None, condition: str
) -> tuple[list[int], list[int]]:
    tokenizer = CharacterTokenizer()
    prefix = (
        row["teacher_prompt_token_ids"]
        if condition.endswith("hinted") and not condition.endswith("unhinted")
        else row["prompt_token_ids"]
    )
    if descriptor is None:
        return prefix, row["completion_token_ids"]
    turn = descriptor["decision_turn"]
    start = row["token_turn_indices"].index(turn)
    raw = row["turn_completions"][str(turn)]
    original = json.dumps(row["turn_commands"][str(turn)][0])[1:-1]
    alternative = json.dumps(descriptor["decision_command"])[1:-1]
    completion = tokenizer.encode(raw.replace(original, alternative, 1))
    return [*prefix, *row["completion_token_ids"][:start]], completion


@pytest.mark.parametrize("condition", ["student_unhinted", "teacher_hinted"])
def test_bounded_worker_reuses_one_real_engine_and_preserves_context(
    tmp_path: Path, condition: str
) -> None:
    episodes_path, descriptors_path, episodes, descriptors = _inputs(tmp_path)
    runtime_ids: list[int] = []
    input_paths: list[Path] = []
    scored_inputs: list[dict[str, Any]] = []

    def real_scorer(**kwargs: Any) -> score.ScoreResult:
        rows = score._read_jsonl(kwargs["samples_path"])
        assert 0 < len(rows) <= 4
        scored_inputs.extend(rows)
        runtime_ids.append(id(kwargs["runtime"]))
        input_paths.append(kwargs["samples_path"])
        assert kwargs["compact_context"] is True
        return score.score_round(
            **kwargs,
            llm_class=CountingTinyTeacherEngine,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )

    output_dir = tmp_path / "scores"
    result = diagnostic_worker.run_worker(
        episodes_path=episodes_path,
        descriptors_path=descriptors_path,
        output_dir=output_dir,
        condition=condition,
        model="synthetic-tiny",
        tokenizer=cast("Any", CharacterTokenizer()),
        chunk_size=4,
        scorer=real_scorer,
    )
    applicable = [
        row
        for row in episodes
        if (row["hint_family"] == "none") == (condition == "student_unhinted")
    ]
    source = {row["key"]: row for row in applicable}
    applicable_descriptors = {
        row["key"]: row for row in descriptors if row["decision_source_key"] in source
    }
    assert result.generated_count == 25
    assert result.resumed_count == 0
    assert result.chunk_count == 7
    assert CountingTinyTeacherEngine.initialized == 1
    assert len(set(runtime_ids)) == 1
    assert all(not path.exists() for path in input_paths)
    outputs = [
        row
        for part in sorted((output_dir / "parts").glob("part-*.jsonl"))
        for row in score._read_jsonl(part)
    ]
    if condition == "teacher_hinted":
        outputs = [*score._read_jsonl(output_dir / "round-0.jsonl"), *outputs]
    assert len(outputs) == len(scored_inputs) == 25
    assert {row["key"] for row in outputs} == {*source, *applicable_descriptors}
    requested = {row["key"]: row for row in scored_inputs}
    for output in outputs:
        descriptor = applicable_descriptors.get(output["key"])
        row = source[descriptor["decision_source_key"] if descriptor else output["key"]]
        prefix, completion = _expected_sequence(row, descriptor, condition)
        requested_prefix = requested[output["key"]].get(
            "teacher_prompt_token_ids", requested[output["key"]]["prompt_token_ids"]
        )
        assert requested_prefix[: len(prefix)] == prefix
        assert [*requested_prefix, *output["completion_token_ids"]] == [*prefix, *completion]
        assert output["context_storage"] == "sha256"
        assert "prompt_token_ids" not in output
        assert "teacher_prompt_token_ids" not in output
        prefix_field = (
            "teacher_prompt_token_ids_sha256"
            if condition == "teacher_hinted"
            else "prompt_token_ids_sha256"
        )
        assert output[prefix_field] == token_ids_sha256(requested_prefix, [])


def test_worker_interruption_reuses_persisted_parts_without_rescoring(tmp_path: Path) -> None:
    episodes_path, descriptors_path, _, _ = _inputs(tmp_path)
    output_dir = tmp_path / "scores"
    calls = 0
    input_paths: list[Path] = []

    def interrupted_scorer(**kwargs: Any) -> score.ScoreResult:
        nonlocal calls
        calls += 1
        input_paths.append(kwargs["samples_path"])
        if calls == 2:
            raise RuntimeError("synthetic worker interruption")
        return score.score_round(
            **kwargs,
            llm_class=CountingTinyTeacherEngine,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )

    kwargs: Any = {
        "episodes_path": episodes_path,
        "descriptors_path": descriptors_path,
        "output_dir": output_dir,
        "condition": "student_unhinted",
        "model": "synthetic-tiny",
        "tokenizer": CharacterTokenizer(),
        "chunk_size": 4,
    }
    with pytest.raises(RuntimeError, match="synthetic worker interruption"):
        diagnostic_worker.run_worker(**kwargs, scorer=interrupted_scorer)
    preserved = {part: part.read_bytes() for part in (output_dir / "parts").glob("part-*.jsonl")}
    assert sum(len(contents.splitlines()) for contents in preserved.values()) == 4
    assert all(not path.exists() for path in input_paths)
    CountingTinyTeacherEngine.initialized = 0
    CountingTinyTeacherEngine.requests = []

    def real_scorer(**score_kwargs: Any) -> score.ScoreResult:
        return score.score_round(
            **score_kwargs,
            llm_class=CountingTinyTeacherEngine,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )

    resumed = diagnostic_worker.run_worker(**kwargs, scorer=real_scorer)
    assert resumed.generated_count == 21
    assert resumed.resumed_count == 4
    assert CountingTinyTeacherEngine.initialized == 1
    assert len(CountingTinyTeacherEngine.requests) == 21
    assert {part: part.read_bytes() for part in preserved} == preserved
    CountingTinyTeacherEngine.requests = []
    complete = diagnostic_worker.run_worker(**kwargs, scorer=real_scorer)
    assert complete.generated_count == 0
    assert complete.resumed_count == 25
    assert CountingTinyTeacherEngine.requests == []


def test_worker_partial_part_resume_keeps_persisted_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes_path, descriptors_path, _, _ = _inputs(tmp_path)
    output_dir = tmp_path / "scores"
    original_append = score._append_jsonl

    def interrupt_after_append(path: Path, row: Any) -> None:
        original_append(path, row)
        raise RuntimeError("synthetic persisted record interruption")

    def real_scorer(**kwargs: Any) -> score.ScoreResult:
        return score.score_round(
            **kwargs,
            llm_class=CountingTinyTeacherEngine,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
            score_batch_size=1,
        )

    kwargs: Any = {
        "episodes_path": episodes_path,
        "descriptors_path": descriptors_path,
        "output_dir": output_dir,
        "condition": "student_unhinted",
        "model": "synthetic-tiny",
        "tokenizer": CharacterTokenizer(),
        "chunk_size": 4,
        "scorer": real_scorer,
    }
    with monkeypatch.context() as interruption:
        interruption.setattr(score, "_append_jsonl", interrupt_after_append)
        with pytest.raises(RuntimeError, match="persisted record interruption"):
            diagnostic_worker.run_worker(**kwargs)
    part = output_dir / "parts" / "part-000000.jsonl"
    preserved = part.read_bytes()
    assert len(preserved.splitlines()) == 1
    CountingTinyTeacherEngine.requests = []
    resumed = diagnostic_worker.run_worker(**kwargs)
    assert resumed.generated_count == 24
    assert resumed.resumed_count == 1
    assert len(CountingTinyTeacherEngine.requests) == 24
    assert part.read_bytes().startswith(preserved)
    completed = [
        row
        for path in sorted((output_dir / "parts").glob("part-*.jsonl"))
        for row in score._read_jsonl(path)
    ]
    assert len(completed) == len({row["key"] for row in completed}) == 25


def test_worker_refuses_chunk_size_change_before_mutating_completed_parts(tmp_path: Path) -> None:
    episodes_path, descriptors_path, _, _ = _inputs(tmp_path)
    output_dir = tmp_path / "scores"

    def real_scorer(**kwargs: Any) -> score.ScoreResult:
        return score.score_round(
            **kwargs,
            llm_class=CountingTinyTeacherEngine,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )

    kwargs: Any = {
        "episodes_path": episodes_path,
        "descriptors_path": descriptors_path,
        "output_dir": output_dir,
        "condition": "student_unhinted",
        "model": "synthetic-tiny",
        "tokenizer": CharacterTokenizer(),
        "scorer": real_scorer,
    }
    diagnostic_worker.run_worker(**kwargs, chunk_size=4)
    preserved = {path: path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}
    CountingTinyTeacherEngine.requests = []
    with pytest.raises(ValueError, match=r"chunk|identity|resume"):
        diagnostic_worker.run_worker(**kwargs, chunk_size=8)
    assert {
        path: path.read_bytes() for path in output_dir.rglob("*") if path.is_file()
    } == preserved
    assert CountingTinyTeacherEngine.requests == []


def test_worker_refuses_descriptor_change_before_mutating_completed_parts(tmp_path: Path) -> None:
    episodes_path, descriptors_path, _, descriptors = _inputs(tmp_path)
    output_dir = tmp_path / "scores"

    def real_scorer(**kwargs: Any) -> score.ScoreResult:
        return score.score_round(
            **kwargs,
            llm_class=CountingTinyTeacherEngine,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )

    kwargs: Any = {
        "episodes_path": episodes_path,
        "descriptors_path": descriptors_path,
        "output_dir": output_dir,
        "condition": "student_unhinted",
        "model": "synthetic-tiny",
        "tokenizer": CharacterTokenizer(),
        "scorer": real_scorer,
    }
    diagnostic_worker.run_worker(**kwargs)
    preserved = {path: path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}
    descriptors[0]["decision_class"] = "synthetic-reclassified"
    _write(descriptors_path, descriptors)
    CountingTinyTeacherEngine.requests = []
    with pytest.raises(ValueError, match=r"descriptor|input|identity|resume"):
        diagnostic_worker.run_worker(**kwargs)
    assert {
        path: path.read_bytes() for path in output_dir.rglob("*") if path.is_file()
    } == preserved
    assert CountingTinyTeacherEngine.requests == []


def test_worker_pins_resolved_revision_and_refuses_changed_cached_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes_path, descriptors_path, _, _ = _inputs(tmp_path)
    revision_calls: list[str] = []
    scored_revisions: list[str] = []
    current_revision = "synthetic-revision-one"

    def resolve_revision(args: Any) -> str:
        revision_calls.append(args.model)
        return current_revision

    def scorer(**kwargs: Any) -> score.ScoreResult:
        scored_revisions.append(kwargs["model_revision"])
        return score.ScoreResult(
            kwargs["output_path"], len(score._read_jsonl(kwargs["samples_path"])), 0
        )

    monkeypatch.setattr(diagnostic_worker, "resolve_model_revision", resolve_revision)
    kwargs: Any = {
        "episodes_path": episodes_path,
        "descriptors_path": descriptors_path,
        "output_dir": tmp_path / "scores",
        "condition": "student_unhinted",
        "model": "synthetic-tiny",
        "tokenizer": CharacterTokenizer(),
        "scorer": scorer,
    }
    diagnostic_worker.run_worker(**kwargs)
    assert revision_calls == ["synthetic-tiny"]
    assert set(scored_revisions) == {"synthetic-revision-one"}
    current_revision = "synthetic-revision-two"
    with pytest.raises(ValueError, match="resume configuration"):
        diagnostic_worker.run_worker(**kwargs)
    assert revision_calls == ["synthetic-tiny", "synthetic-tiny"]
