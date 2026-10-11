"""Synthetic token-only tests for separate teacher-prefix scoring."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.distill.score import score_round
from reward_hacking.tests.test_distill_sample_score import FakeSamplingParams, FakeTorch

if TYPE_CHECKING:
    from pathlib import Path


class PositionScoreEngine:
    instances: ClassVar[list[PositionScoreEngine]] = []
    invalid_logp: ClassVar[float | None] = None
    extra_row: ClassVar[bool] = False
    wrong_returned_ids: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.prompts: list[dict[str, list[int]]] = []
        self.instances.append(self)

    def generate(self, prompts: list[dict[str, list[int]]], **_kwargs: Any) -> list[Any]:
        self.prompts.extend(prompts)
        outputs = []
        for prompt in prompts:
            rows: list[Any] = [None]
            for position, token_id in enumerate(prompt["prompt_token_ids"][1:], start=1):
                rows.append(
                    {
                        **{
                            index: SimpleNamespace(logprob=-index / 10, rank=index)
                            for index in range(1, 5)
                        },
                        token_id: SimpleNamespace(
                            logprob=-float(position)
                            if self.invalid_logp is None
                            else self.invalid_logp,
                            rank=5,
                        ),
                    }
                )
            if self.extra_row:
                rows.append(rows[-1])
            outputs.append(
                SimpleNamespace(
                    prompt_logprobs=rows,
                    prompt_token_ids=[999]
                    if self.wrong_returned_ids
                    else prompt["prompt_token_ids"],
                )
            )
        return outputs


@pytest.fixture(autouse=True)
def reset_engine() -> None:
    PositionScoreEngine.instances.clear()
    PositionScoreEngine.invalid_logp = None
    PositionScoreEngine.extra_row = False
    PositionScoreEngine.wrong_returned_ids = False


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    samples = tmp_path / "samples.jsonl"
    teachers = tmp_path / "prompts.jsonl"
    output = tmp_path / "scores.jsonl"
    write_rows(
        samples,
        [
            {
                "key": "synthetic-sample",
                "env": "synthetic",
                "problem_id": "synthetic-id",
                "prompt_token_ids": [80, 81],
                "completion_token_ids": [90, 91],
                "student_sampling_logps": [-0.5, -0.6],
                "token_ids_sha256": token_ids_sha256([80, 81], [90, 91]),
            }
        ],
    )
    write_rows(
        teachers,
        [{"env": "synthetic", "problem_id": "synthetic-id", "prompt_token_ids": [70, 71, 72]}],
    )
    return samples, teachers, output


def run_score(paths: tuple[Path, Path, Path], **kwargs: Any) -> Any:
    samples, teachers, output = paths
    return score_round(
        samples_path=samples,
        teacher_prompts_path=teachers,
        output_path=output,
        model="synthetic-model",
        llm_class=PositionScoreEngine,
        sampling_params_class=FakeSamplingParams,
        torch_module=FakeTorch(),
        **kwargs,
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("prefix", [[70], [70, 71, 72, 73]])
def test_exact_completion_ids_with_different_teacher_boundary(
    tmp_path: Path, prefix: list[int]
) -> None:
    paths = inputs(tmp_path)
    write_rows(
        paths[1], [{"env": "synthetic", "problem_id": "synthetic-id", "prompt_token_ids": prefix}]
    )
    assert run_score(paths).generated_count == 1
    assert PositionScoreEngine.instances[0].prompts == [{"prompt_token_ids": [*prefix, 90, 91]}]
    record = read_rows(paths[2])[0]
    assert record["teacher_logps"] == [-float(len(prefix)), -float(len(prefix) + 1)]
    assert record["token_ids_sha256"] == token_ids_sha256([80, 81], [90, 91])
    assert record["teacher_token_ids_sha256"] == token_ids_sha256(prefix, [90, 91])
    assert len(record["scoring_identity"]) == 64
    assert run_score(paths).resumed_count == 1
    assert len(PositionScoreEngine.instances) == 1


@pytest.mark.parametrize("change", ["prompt", "engine", "legacy", "hash", "scores"])
def test_resume_checks_context_and_scores_but_accepts_missing_provenance(
    tmp_path: Path, change: str
) -> None:
    paths = inputs(tmp_path)
    run_score(paths)
    kwargs: dict[str, Any] = {}
    if change == "prompt":
        write_rows(
            paths[1], [{"env": "synthetic", "problem_id": "synthetic-id", "prompt_token_ids": [70]}]
        )
    elif change == "engine":
        kwargs["kv_cache_dtype"] = "fp8"
    else:
        records = read_rows(paths[2])
        if change == "legacy":
            records[0].pop("scoring_identity")
        elif change == "hash":
            records[0]["teacher_token_ids_sha256"] = "incorrect"
        else:
            records[0]["teacher_logps"] = [float("nan"), -1.0]
        write_rows(paths[2], records)
    if change == "legacy":
        result = run_score(paths, **kwargs)
        assert (result.generated_count, result.resumed_count) == (0, 1)
    else:
        expected_error = {
            "prompt": "teacher context tokens differ",
            "engine": "scoring settings differ",
            "hash": "teacher_token_ids_sha256",
            "scores": "finite",
        }[change]
        with pytest.raises(ValueError, match=expected_error):
            run_score(paths, **kwargs)
    assert len(PositionScoreEngine.instances) == 1


@pytest.mark.parametrize("prefix", [[], [-1], [True], [1.5], "tokens"])
def test_invalid_teacher_ids_refuse_before_engine(tmp_path: Path, prefix: Any) -> None:
    paths = inputs(tmp_path)
    write_rows(
        paths[1], [{"env": "synthetic", "problem_id": "synthetic-id", "prompt_token_ids": prefix}]
    )
    with pytest.raises(
        ValueError, match=r"prompt token ids|prompt_token_ids|identity|env and problem_id"
    ):
        run_score(paths)
    assert not PositionScoreEngine.instances


@pytest.mark.parametrize("change", ["missing", "duplicate", "sample-identity"])
def test_corpus_identity_validation(tmp_path: Path, change: str) -> None:
    paths = inputs(tmp_path)
    rows = read_rows(paths[1])
    if change == "missing":
        rows[0]["problem_id"] = "other-id"
    elif change == "duplicate":
        rows *= 2
    else:
        samples = read_rows(paths[0])
        samples[0].pop("problem_id")
        write_rows(paths[0], samples)
    write_rows(paths[1], rows)
    with pytest.raises(
        ValueError, match=r"prompt token ids|prompt_token_ids|identity|env and problem_id"
    ):
        run_score(paths)
    assert not PositionScoreEngine.instances


@pytest.mark.parametrize("logp", [float("nan"), float("inf"), 0.1])
def test_invalid_returned_logps_refuse(tmp_path: Path, logp: float) -> None:
    paths = inputs(tmp_path)
    PositionScoreEngine.invalid_logp = logp
    with pytest.raises(ValueError, match="finite nonpositive"):
        run_score(paths)
    assert not paths[2].exists() or not paths[2].read_text(encoding="utf-8")


def test_returned_rows_must_align_exactly(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    PositionScoreEngine.extra_row = True
    with pytest.raises(ValueError, match="prompt logprob rows"):
        run_score(paths)


@pytest.mark.parametrize("logps", [[float("nan"), -0.5], [0.1, -0.5], [True, -0.5], [-0.5]])
def test_sampled_logps_must_be_finite_nonpositive_and_aligned(
    tmp_path: Path, logps: list[object]
) -> None:
    paths = inputs(tmp_path)
    rows = read_rows(paths[0])
    rows[0]["student_sampling_logps"] = logps
    write_rows(paths[0], rows)
    with pytest.raises(ValueError, match=r"student_sampling_logps must (contain|align)"):
        run_score(paths)
    assert not PositionScoreEngine.instances


def test_batch_uses_each_teacher_prefix_and_resumes_only_missing_samples(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    samples = read_rows(paths[0])
    sample = dict(samples[0], key="another-synthetic-sample", problem_id="another-synthetic-id")
    samples.append(sample)
    write_rows(paths[0], samples)
    teachers = read_rows(paths[1])
    teachers.append(
        {"env": "synthetic", "problem_id": "another-synthetic-id", "prompt_token_ids": [60]}
    )
    write_rows(paths[1], teachers)
    result = run_score(paths)
    assert result.generated_count == 2
    assert PositionScoreEngine.instances[0].prompts == [
        {"prompt_token_ids": [70, 71, 72, 90, 91]},
        {"prompt_token_ids": [60, 90, 91]},
    ]
    scores = read_rows(paths[2])
    assert [row["teacher_logps"] for row in scores] == [[-3.0, -4.0], [-1.0, -2.0]]
    write_rows(paths[2], scores[:1])
    resumed = run_score(paths)
    assert (resumed.generated_count, resumed.resumed_count) == (1, 1)
    assert PositionScoreEngine.instances[-1].prompts == [{"prompt_token_ids": [60, 90, 91]}]
    assert [row["key"] for row in read_rows(paths[2])] == [row["key"] for row in samples]


def test_returned_request_ids_must_match_teacher_forced_sequence(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    PositionScoreEngine.wrong_returned_ids = True
    with pytest.raises(ValueError, match="returned token ids"):
        run_score(paths)
