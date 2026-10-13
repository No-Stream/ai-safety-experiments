"""Synthetic token-only tests for separate teacher-prefix scoring."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.distill.score import compute_kl_summary, score_round
from reward_hacking.tests.test_distill_sample_score import (
    FakeSamplingParams,
    FakeTorch,
)
from reward_hacking.tests.test_distill_sample_score import (
    fake_gpu_query as fake_gpu_query,  # noqa: PLC0414 - registers pytest fixture
)

if TYPE_CHECKING:
    from pathlib import Path


class PositionScoreEngine:
    instances: ClassVar[list[PositionScoreEngine]] = []
    invalid_logp: ClassVar[float | None] = None
    extra_row: ClassVar[bool] = False
    wrong_returned_ids: ClassVar[bool | None] = False
    corrupt_position: ClassVar[int | None] = None
    missing_actual: ClassVar[bool] = False

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
            if self.corrupt_position is not None:
                rows[self.corrupt_position] = None
            if self.missing_actual:
                rows[-1].pop(prompt["prompt_token_ids"][-1])
            if self.extra_row:
                rows.append(rows[-1])
            outputs.append(
                SimpleNamespace(
                    prompt_logprobs=rows,
                    prompt_token_ids=None
                    if self.wrong_returned_ids is None
                    else [999]
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
    PositionScoreEngine.corrupt_position = None
    PositionScoreEngine.missing_actual = False


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
    if change in {"legacy", "hash"}:
        result = run_score(paths, **kwargs)
        assert (result.generated_count, result.resumed_count) == (0, 1)
    else:
        expected_error = {
            "prompt": "teacher context tokens differ",
            "engine": "scoring settings differ",
            "scores": "finite",
        }[change]
        with pytest.raises(ValueError, match=expected_error):
            run_score(paths, **kwargs)
    assert len(PositionScoreEngine.instances) == 1


def test_resume_compares_recorded_teacher_completion_tokens(tmp_path: Path) -> None:
    samples, _teachers, output = inputs(tmp_path)
    sample = read_rows(samples)[0]
    sample.update(
        teacher_prompt_token_ids=[70, 71],
        teacher_completion_token_ids=[90, 55, 91],
        teacher_token_positions=[2, 4],
    )
    write_rows(samples, [sample])

    def score() -> Any:
        return score_round(
            samples_path=samples,
            output_path=output,
            model="synthetic-model",
            llm_class=PositionScoreEngine,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )

    assert score().generated_count == 1
    record = read_rows(output)[0]
    assert record["teacher_completion_token_ids"] == [90, 55, 91]
    assert record["teacher_logps"] == [-2.0, -4.0]
    assert score().resumed_count == 1

    sample["teacher_completion_token_ids"] = [90, 56, 91]
    write_rows(samples, [sample])
    with pytest.raises(ValueError, match="teacher sequence tokens differ"):
        score()
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


@pytest.mark.parametrize("prefix", [[70], [70, 71, 72, 73]])
def test_masked_full_sequence_uses_per_row_teacher_prefix(
    tmp_path: Path, prefix: list[int]
) -> None:
    paths = inputs(tmp_path)
    rows = read_rows(paths[0])
    rows[0].update(
        teacher_prompt_token_ids=prefix,
        completion_token_ids=[90, 91, 92, 93],
        loss_mask=[1, 0, 0, 1],
        student_sampling_logps=[-0.5, None, None, -0.6],
        token_ids_sha256=token_ids_sha256([80, 81], [90, 91, 92, 93]),
    )
    write_rows(paths[0], rows)
    result = score_round(
        samples_path=paths[0],
        output_path=paths[2],
        model="synthetic-model",
        llm_class=PositionScoreEngine,
        sampling_params_class=FakeSamplingParams,
        torch_module=FakeTorch(),
    )
    assert result.generated_count == 1
    assert PositionScoreEngine.instances[0].prompts == [
        {"prompt_token_ids": [*prefix, 90, 91, 92, 93]}
    ]
    record = read_rows(paths[2])[0]
    assert record["teacher_logps"] == [-float(len(prefix)), 0.0, 0.0, -float(len(prefix) + 3)]
    assert record["top_ids"][1:3] == [[], []]
    assert record["loss_mask"] == [1, 0, 0, 1]
    assert record["teacher_prompt_token_ids"] == prefix
    assert record["teacher_token_ids_sha256"] == token_ids_sha256(prefix, [90, 91, 92, 93])


@pytest.mark.parametrize("mask", [[0, 0], [1], [True, 0], [1, 2], None])
def test_invalid_completion_mask_refuses_before_scoring(tmp_path: Path, mask: object) -> None:
    paths = inputs(tmp_path)
    rows = read_rows(paths[0])
    rows[0]["loss_mask"] = mask
    write_rows(paths[0], rows)
    with pytest.raises(ValueError, match="loss_mask"):
        run_score(paths)
    assert not PositionScoreEngine.instances


@pytest.mark.parametrize("returned_ids", [True, None])
def test_returned_request_ids_are_checked_without_separate_teacher_corpus(
    tmp_path: Path, returned_ids: bool | None
) -> None:
    paths = inputs(tmp_path)
    PositionScoreEngine.wrong_returned_ids = returned_ids
    with pytest.raises(ValueError, match="returned token ids"):
        score_round(
            samples_path=paths[0],
            output_path=paths[2],
            model="synthetic-model",
            llm_class=PositionScoreEngine,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )


@pytest.mark.parametrize("missing_actual", [True, False])
def test_active_completion_requires_actual_token_row(tmp_path: Path, missing_actual: bool) -> None:
    paths = inputs(tmp_path)
    PositionScoreEngine.missing_actual = missing_actual
    PositionScoreEngine.corrupt_position = None if missing_actual else 3
    with pytest.raises(ValueError, match=r"omitted.*(token|logprobs)"):
        run_score(paths)


def test_environment_logprob_row_is_ignored_but_context_is_kept(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    rows = read_rows(paths[0])
    rows[0].update(loss_mask=[0, 1], student_sampling_logps=[None, -0.6])
    write_rows(paths[0], rows)
    PositionScoreEngine.corrupt_position = 3
    run_score(paths)
    assert PositionScoreEngine.instances[0].prompts == [{"prompt_token_ids": [70, 71, 72, 90, 91]}]
    assert read_rows(paths[2])[0]["teacher_logps"] == [0.0, -4.0]


def test_masked_summary_counts_only_assistant_logprob_gaps(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    samples = read_rows(paths[0])
    samples[0].update(loss_mask=[0, 1], student_sampling_logps=[None, -0.6])
    write_rows(paths[0], samples)
    run_score(paths)
    assert compute_kl_summary(paths[0], paths[2])["synthetic"] == {
        "mean_student_sampling_logp_minus_teacher_logp": 3.4,
        "token_count": 1,
        "sample_count": 1,
    }


@pytest.mark.parametrize("change", ["mask", "prefix"])
def test_masked_resume_refuses_changed_teacher_tokens_or_mask(tmp_path: Path, change: str) -> None:
    paths = inputs(tmp_path)
    samples = read_rows(paths[0])
    samples[0].update(teacher_prompt_token_ids=[70], loss_mask=[1, 0])
    write_rows(paths[0], samples)
    kwargs: dict[str, Any] = {
        "samples_path": paths[0],
        "output_path": paths[2],
        "model": "synthetic-model",
        "llm_class": PositionScoreEngine,
        "sampling_params_class": FakeSamplingParams,
        "torch_module": FakeTorch(),
    }
    score_round(**kwargs)
    assert score_round(**kwargs).resumed_count == 1
    samples[0]["loss_mask" if change == "mask" else "teacher_prompt_token_ids"] = (
        [0, 1] if change == "mask" else [71]
    )
    write_rows(paths[0], samples)
    with pytest.raises(ValueError, match=r"loss_mask|teacher context"):
        score_round(**kwargs)


def test_sample_hash_is_provenance_and_is_recomputed_from_actual_tokens(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    samples = read_rows(paths[0])
    samples[0]["token_ids_sha256"] = "stale-provenance"
    write_rows(paths[0], samples)
    run_score(paths)
    assert read_rows(paths[2])[0]["token_ids_sha256"] == token_ids_sha256([80, 81], [90, 91])


def test_completed_scoring_resume_allows_different_batch_capacity(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    assert run_score(paths, max_num_seqs=4).generated_count == 1
    assert run_score(paths, max_num_seqs=1).resumed_count == 1
    assert len(PositionScoreEngine.instances) == 1
