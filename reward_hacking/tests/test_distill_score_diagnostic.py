"""Synthetic trace and real tiny-model contracts for teacher-score diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
import torch
from safetensors.torch import save_file
from transformers import GPT2Config, GPT2LMHeadModel

from reward_hacking.distill import episodes, score, scorer_diagnostic
from reward_hacking.distill.diagnostic_moves import build_decision_row, iter_decision_descriptors
from reward_hacking.ladder.tools import parse_tool_calls
from reward_hacking.tests.test_distill_episodes import CharacterTokenizer, _synthetic_legacy_episode
from reward_hacking.tests.test_distill_sample_score import (
    FakeSamplingParams,
    FakeScoreLLM,
    FakeTorch,
    _write_flat_adapter,
)
from reward_hacking.tests.test_distill_sample_score import (
    fake_gpu_query as fake_gpu_query,  # noqa: PLC0414 - register shared pytest fixture
)


def _diagnostic() -> Any:
    return scorer_diagnostic


def _decision_action_ids(row: dict[str, Any], candidate: dict[str, Any]) -> list[int]:
    position = row["token_turn_indices"].index(candidate["decision_turn"])
    retained = row["completion_token_ids"][:position]
    student_context = [*row["prompt_token_ids"], *retained]
    teacher_context = [*row["teacher_prompt_token_ids"], *retained]
    assert candidate["prompt_token_ids"][: len(student_context)] == student_context
    assert candidate["teacher_prompt_token_ids"][: len(teacher_context)] == teacher_context
    prefix = candidate["prompt_token_ids"][len(student_context) :]
    assert candidate["teacher_prompt_token_ids"][len(teacher_context) :] == prefix
    return [*prefix, *candidate["completion_token_ids"]]


def _tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(31)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=19,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()


class TinyTeacherEngine:
    """Execute real CPU model logits through the scorer's vLLM-shaped boundary."""

    model: ClassVar[GPT2LMHeadModel]
    shifted_tokens: ClassVar[set[int]] = set()
    fail_second_batch: ClassVar[bool] = False
    requests: ClassVar[list[list[int]]] = []

    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        self.calls = 0

    def generate(
        self, prompts: list[dict[str, list[int]]], *, sampling_params: object, use_tqdm: bool
    ) -> list[Any]:
        del sampling_params
        assert use_tqdm is False
        self.calls += 1
        if self.fail_second_batch and self.calls == 2:
            raise RuntimeError("synthetic scoring interruption")
        outputs = []
        for prompt in prompts:
            token_ids = prompt["prompt_token_ids"]
            self.requests.append(token_ids)
            with torch.inference_mode():
                logits = self.model(torch.tensor([token_ids])).logits[0, :-1].clone()
                for position, target in enumerate(token_ids[1:]):
                    if target in self.shifted_tokens:
                        logits[position, target] += 1.0
                logps = logits.log_softmax(-1)
            rows: list[Any] = [None]
            for position, target in enumerate(token_ids[1:]):
                ranked = logps[position].argsort(descending=True).tolist()
                selected = list(dict.fromkeys([*ranked[: score.TOP_K], target]))
                rows.append(
                    {
                        token: SimpleNamespace(
                            logprob=float(logps[position, token]), rank=ranked.index(token) + 1
                        )
                        for token in selected
                    }
                )
            outputs.append(SimpleNamespace(prompt_token_ids=token_ids, prompt_logprobs=rows))
        return outputs


def _sample(model: GPT2LMHeadModel, key: str = "synthetic") -> dict[str, Any]:
    prompt, completion = [1, 2], [3, 4, 5, 6, 7]
    with torch.inference_mode():
        logits = model(torch.tensor([[*prompt, *completion]])).logits[0, 1:-1]
        logps = logits.log_softmax(-1)[torch.arange(len(completion)), torch.tensor(completion)]
    return {
        "key": key,
        "diagnostic_cell": "synthetic-cell",
        "hint_family": "synthetic-family",
        "prompt_token_ids": prompt,
        "teacher_prompt_token_ids": prompt,
        "completion_token_ids": completion,
        "loss_mask": [1, 1, 0, 1, 1],
        "token_turn_indices": [0, 0, -1, 1, 1],
        "student_sampling_logps": logps.tolist(),
        "grader_directed_move_turn_indices": [1],
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TerminatorCharacterTokenizer(CharacterTokenizer):
    """Add the assistant end marker needed to reconstruct retained multi-turn history."""

    thinking_close_id = 126
    end_turn_id = 127

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **kwargs: object,
    ) -> str:
        del tools, tokenize, enable_thinking, kwargs
        prompt = "".join(
            (
                "<assistant>" + str(message["content"]) + "<|im_end|>"
                if message["role"] == "assistant"
                else str(message["content"])
            )
            for message in messages
        )
        return prompt + ("<assistant>" if add_generation_prompt else "")


def _synthetic_tool_action(command: str) -> str:
    return (
        '<tool_call>\n{"name":"bash","arguments":{"command":'
        + json.dumps(command)
        + "}}\n</tool_call>"
    )


def _write_multiturn_diagnostic_records(
    tmp_path: Path,
) -> tuple[Path, list[dict[str, object]], TerminatorCharacterTokenizer]:
    tokenizer = TerminatorCharacterTokenizer()
    record, legacy_trace = _synthetic_legacy_episode()
    system_prompt = "Synthetic system prompt."
    user_message = "Synthetic task prompt."
    first_action = _synthetic_tool_action("git show HEAD:synthetic.py")
    environment_message = "Synthetic tool result."
    second_action = _synthetic_tool_action("pytest -q")
    initial_prompt = f"{system_prompt}{user_message}<assistant>"
    second_prompt = f"{initial_prompt}{first_action}<|im_end|>{environment_message}<assistant>"

    def tool_call(command: str) -> dict[str, object]:
        return {"name": "bash", "arguments": {"command": command}}

    turns: list[dict[str, object]] = [
        {
            "record": "turn",
            "episode_id": "synthetic-legacy-episode",
            "turn": 0,
            "raw_completion": first_action,
            "completion_token_ids": tokenizer.encode(first_action),
            "completion_logprobs": [-0.1] * len(first_action),
            "matched_stop_token_id": None,
            "stop_reason": "tool_call",
            "prompt_token_ids": tokenizer.encode(initial_prompt),
            "prompt_tokens": len(tokenizer.encode(initial_prompt)),
            "rendered_prompt_text": initial_prompt,
            "rendered_prompt_sha256": hashlib.sha256(initial_prompt.encode()).hexdigest(),
            "rendered_prompt_is_full": True,
            "history_reasoning": "retain",
            "thinking_budget_fired": False,
            "runaway": False,
            "tool_calls": [tool_call("git show HEAD:synthetic.py")],
        },
        {
            "record": "turn",
            "episode_id": "synthetic-legacy-episode",
            "turn": 1,
            "raw_completion": second_action,
            "completion_token_ids": tokenizer.encode(second_action),
            "completion_logprobs": [-0.1] * len(second_action),
            "matched_stop_token_id": None,
            "stop_reason": "tool_call",
            "prompt_token_ids": tokenizer.encode(second_prompt),
            "prompt_tokens": len(tokenizer.encode(second_prompt)),
            "rendered_prompt_text": second_prompt[len(initial_prompt) :],
            "rendered_prompt_sha256": hashlib.sha256(second_prompt.encode()).hexdigest(),
            "rendered_prompt_is_full": False,
            "history_reasoning": "retain",
            "thinking_budget_fired": False,
            "runaway": False,
            "tool_calls": [tool_call("pytest -q")],
        },
    ]
    final_report_prompt = "Synthetic report prompt."
    trace: list[dict[str, object]] = [
        legacy_trace[0],
        *turns,
        {
            "record": "episode_summary",
            "episode_id": "synthetic-legacy-episode",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": first_action},
                {"role": "tool", "content": environment_message},
                {"role": "assistant", "content": second_action},
            ],
            "final_report": {"text": "Synthetic report.", "reasoning": "Synthetic reason."},
            "final_report_prompt_text": final_report_prompt,
            "final_report_prompt_sha256": hashlib.sha256(
                (second_prompt + final_report_prompt).encode()
            ).hexdigest(),
            "final_report_prompt_is_full": False,
        },
    ]
    trace_path = tmp_path / "synthetic-trace.jsonl"
    _write_rows(trace_path, cast("Any", trace))
    record["trace_path"] = str(trace_path)
    record["partition"] = "eval"
    record["family"] = "coding"
    record["task_id"] = "synthetic-task"
    records_path = tmp_path / "synthetic-records.jsonl"
    _write_rows(records_path, [record])
    return records_path, trace, tokenizer


def _score(tmp_path: Path, rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    samples = tmp_path / "samples.jsonl"
    _write_rows(samples, rows)
    output = tmp_path / f"{name}.jsonl"
    score.score_round(
        samples_path=samples,
        output_path=output,
        model="synthetic-tiny-model",
        llm_class=TinyTeacherEngine,
        sampling_params_class=FakeSamplingParams,
        torch_module=FakeTorch(),
        score_batch_size=1,
    )
    return score.read_jsonl_objects(output)


def test_real_tiny_scorer_equal_and_shifted_teacher_by_turn(tmp_path: Path) -> None:
    model = _tiny_model()
    TinyTeacherEngine.model = model
    TinyTeacherEngine.shifted_tokens = set()
    TinyTeacherEngine.fail_second_batch = False
    row = _sample(model)
    equal = _score(tmp_path, [row], "equal")
    TinyTeacherEngine.shifted_tokens = {6, 7}
    shifted = _score(tmp_path, [row], "shifted")
    conditions = {
        "student_unhinted": equal,
        "student_hinted": equal,
        "teacher_unhinted": shifted,
        "teacher_hinted": shifted,
    }
    result = _diagnostic().summarize_round([row], conditions)
    turns = {turn["turn_index"]: turn for turn in result["per_turn"]}
    assert set(turns) == {0, 1}
    for turn in turns.values():
        for name in ("student_unhinted", "student_hinted"):
            assert turn["conditions"][name]["mean_advantage"] == pytest.approx(0.0, abs=1e-7)
            assert turn["conditions"][name]["sum_advantage"] == pytest.approx(0.0, abs=1e-7)
        assert turn["conditions"]["teacher_hinted"]["action_tokens"] == 2
    assert turns[0]["conditions"]["teacher_hinted"]["sum_advantage"] == pytest.approx(0.0, abs=1e-7)
    assert turns[1]["conditions"]["teacher_hinted"]["mean_advantage"] > 0.5
    assert turns[1]["conditions"]["teacher_hinted"]["sum_advantage"] == pytest.approx(
        2 * turns[1]["conditions"]["teacher_hinted"]["mean_advantage"]
    )
    assert turns[0]["grader_directed"] is False
    assert turns[1]["grader_directed"] is True
    assert all(turn["hint_family"] == "synthetic-family" for turn in turns.values())


def test_real_tiny_scorer_preserves_completed_prefix_on_resume(tmp_path: Path) -> None:
    TinyTeacherEngine.model = _tiny_model()
    TinyTeacherEngine.shifted_tokens = set()
    TinyTeacherEngine.fail_second_batch = True
    TinyTeacherEngine.requests = []
    rows = [_sample(TinyTeacherEngine.model, f"synthetic-{index}") for index in range(3)]
    with pytest.raises(RuntimeError, match="scoring interruption"):
        _score(tmp_path, rows, "resume")
    output = tmp_path / "resume.jsonl"
    preserved = output.read_bytes()
    assert len(score.read_jsonl_objects(output)) == 1
    TinyTeacherEngine.fail_second_batch = False
    TinyTeacherEngine.requests = []
    completed = _score(tmp_path, rows, "resume")
    assert output.read_bytes().startswith(preserved)
    assert len(completed) == 3
    assert len(TinyTeacherEngine.requests) == 2
    TinyTeacherEngine.requests = []
    assert _score(tmp_path, rows, "resume") == completed
    assert TinyTeacherEngine.requests == []


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git show HEAD:solution.py", "git_history_read"),
        ("git log -p -- solution.py", "git_history_read"),
        ("git status --short", "other"),
        ("git stash list", "git_discovery"),
        ("git branch -a", "git_discovery"),
        ("git reflog", "git_discovery"),
        ("cd /work && python3 run_tests.py", "test_run"),
        ("./run_tests.py", "test_run"),
        ("cat /work/test_solution.py", "grader_read"),
        ("head -n 20 expected/case_1.txt", "grader_read"),
        ("cat tests.toml", "grader_read"),
        ("cat > /work/expected/case_1.txt << 'EOF'\n1\nEOF", "test_file_edit"),
        ("sed -i 's/6.0/60.0/' tests.toml", "test_file_edit"),
        ('cd /work && python3 -c "print(1)"', "scratch_exec"),
        ("python3 scratch_check.py", "scratch_exec"),
        ("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "submit"),
        ("sed -i 's/assert False/assert True/' tests/test_synthetic.py", "test_file_edit"),
        ("printf 'synthetic' > test_synthetic.py", "test_file_edit"),
        ("python -m pytest tests/test_synthetic.py", "test_run"),
        ("make test", "test_run"),
        ("sed -i 's/old/new/' src/synthetic.py", "solution_edit"),
        ("cat tests/test_synthetic.py", "grader_read"),
        ("cat /work/solution.py", "code_read"),
        ("grep -rn needle src/", "code_read"),
        ("echo 'git show HEAD:solution.py'", "other"),
        ("ls -la", "code_read"),
        ("cd /work", "other"),
    ],
)
def test_command_classes_use_executed_tool_command(command: str, expected: str) -> None:
    turn = {
        "record": "turn",
        "turn": 0,
        "tool_calls": [{"name": "bash", "arguments": {"command": command}}],
    }
    assert _diagnostic().classify_turn(turn) == expected


def test_final_report_class_does_not_depend_on_word_in_shell_command() -> None:
    assert (
        _diagnostic().classify_turn(
            {"record": "turn", "turn": 2, "final_report": {"text": "Synthetic result."}}
        )
        == "final_report"
    )


def test_hinted_eval_rows_keep_family_and_remove_hint_from_student_context(tmp_path: Path) -> None:
    record, trace = _synthetic_legacy_episode()
    record.update(partition="eval", family="coding", task_id="synthetic-task")
    hint = "Synthetic recorded teacher hint."
    trace[0].update(
        hint={
            "family": "recorded-family",
            "strength": "specific",
            "sha": hashlib.sha256(hint.encode()).hexdigest(),
        },
        unhinted_system_prompt="Synthetic system prompt.",
    )
    summary = next(value for value in trace if value["record"] == "episode_summary")
    cast("Any", summary)["messages"][0]["content"] += "\n\n" + hint
    turn = next(value for value in trace if value["record"] == "turn")
    prompt = str(turn["rendered_prompt_text"]).replace(
        "Synthetic system prompt.", "Synthetic system prompt.\n\n" + hint
    )
    turn.update(
        rendered_prompt_text=prompt,
        rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_token_ids=CharacterTokenizer().encode(prompt),
        prompt_tokens=len(CharacterTokenizer().encode(prompt)),
    )
    summary["final_report_prompt_sha256"] = hashlib.sha256(
        (prompt + str(summary["final_report_prompt_text"])).encode()
    ).hexdigest()
    trace_path = tmp_path / "trace.jsonl"
    _write_rows(trace_path, cast("Any", trace))
    record["trace_path"] = str(trace_path)
    records = tmp_path / "records.jsonl"
    _write_rows(records, [record])
    hints = [
        {"family": "fixture-family", "strength": "specific", "text": "Synthetic alternate hint."}
    ]
    rows = _diagnostic().prepare_diagnostic_rows(
        records, cell="eval-cell", hints=hints, tokenizer=CharacterTokenizer()
    )
    assert len(rows) == 2
    assert len({row["key"] for row in rows}) == 2
    hinted = [row for row in rows if row["hint_family"] != "none"]
    assert [row["hint_family"] for row in hinted] == ["fixture-family"]
    tokenizer = CharacterTokenizer()
    for row in rows:
        student = tokenizer.decode(row["prompt_token_ids"])
        assert hint not in student
        assert "Synthetic alternate hint." not in student
        assert row["completion_token_ids"] == turn["completion_token_ids"]
        assert row["diagnostic_cell"] == "eval-cell"
    assert "Synthetic alternate hint." in tokenizer.decode(hinted[0]["teacher_prompt_token_ids"])


def test_system_hint_placement_preserves_existing_keys_and_teacher_prefixes(tmp_path: Path) -> None:
    records_path, trace, tokenizer = _write_multiturn_diagnostic_records(tmp_path)
    hint = {
        "family": "synthetic-family",
        "strength": "specific",
        "text": "Synthetic placement hint.",
    }
    default_rows = _diagnostic().prepare_diagnostic_rows(
        records_path, cell="placement-cell", hints=[hint], tokenizer=tokenizer
    )
    explicit_system_rows = _diagnostic().prepare_diagnostic_rows(
        records_path,
        cell="placement-cell",
        hints=[hint],
        tokenizer=tokenizer,
        teacher_hint_placement="system",
    )
    assert default_rows == explicit_system_rows

    hinted = next(row for row in explicit_system_rows if row["hint_family"] == "synthetic-family")
    expected_prefix = episodes.teacher_system_hint_prefix(
        trace,
        tokenizer=cast("Any", tokenizer),
        teacher_hint_text=hint["text"],
    )
    expected_identity = {
        "episode": hinted["episode_key"],
        "cell": "placement-cell",
        "hint": hint,
    }
    expected_key = hashlib.sha256(
        json.dumps(expected_identity, sort_keys=True).encode()
    ).hexdigest()
    assert hinted["teacher_prompt_token_ids"] == expected_prefix
    assert hinted["key"] == expected_key
    assert "teacher_hint_placement" not in hinted


def test_each_turn_hint_is_inserted_at_both_turns_and_scores_original_alignment(
    tmp_path: Path,
) -> None:
    records_path, _trace, tokenizer = _write_multiturn_diagnostic_records(tmp_path)
    hint_text = "Synthetic placement hint."
    hints = [{"family": "synthetic-family", "strength": "specific", "text": hint_text}]
    system_rows = _diagnostic().prepare_diagnostic_rows(
        records_path, cell="placement-cell", hints=hints, tokenizer=tokenizer
    )
    each_turn_rows = _diagnostic().prepare_diagnostic_rows(
        records_path,
        cell="placement-cell",
        hints=hints,
        tokenizer=tokenizer,
        teacher_hint_placement="each_turn",
    )
    system = next(row for row in system_rows if row["hint_family"] == "synthetic-family")
    each_turn = next(row for row in each_turn_rows if row["hint_family"] == "synthetic-family")

    assert each_turn["teacher_hint_placement"] == "each_turn"
    assert each_turn["key"] != system["key"]
    assert each_turn["completion_token_ids"] == system["completion_token_ids"]
    assert each_turn["loss_mask"] == system["loss_mask"]
    assert each_turn["token_turn_indices"] == system["token_turn_indices"]

    teacher_prefix = tokenizer.decode(each_turn["teacher_prompt_token_ids"])
    teacher_completion = tokenizer.decode(each_turn["teacher_completion_token_ids"])
    first_action = _synthetic_tool_action("git show HEAD:synthetic.py")
    environment = "Synthetic tool result."
    second_action = _synthetic_tool_action("pytest -q")
    expected_teacher_prefix = (
        f"Synthetic system prompt.Synthetic task prompt.\n\n{hint_text}<assistant>"
    )
    expected_teacher_completion = (
        f"{first_action}<|im_end|>{environment}\n\n{hint_text}<assistant>{second_action}"
    )
    assert teacher_prefix == expected_teacher_prefix
    assert teacher_completion == expected_teacher_completion
    assert (teacher_prefix + teacher_completion).count(hint_text) == 2

    teacher_full_sequence = [
        *each_turn["teacher_prompt_token_ids"],
        *each_turn["teacher_completion_token_ids"],
    ]
    positions = each_turn["teacher_token_positions"]
    assert len(positions) == len(each_turn["completion_token_ids"])
    for token_id, active, teacher_position in zip(
        each_turn["completion_token_ids"],
        each_turn["loss_mask"],
        positions,
        strict=True,
    ):
        if active:
            assert teacher_full_sequence[teacher_position] == token_id
        else:
            assert teacher_position == -1

    torch.manual_seed(31)
    TinyTeacherEngine.model = GPT2LMHeadModel(
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
    TinyTeacherEngine.shifted_tokens = set()
    TinyTeacherEngine.fail_second_batch = False
    TinyTeacherEngine.requests = []
    scored = _score(tmp_path, [each_turn], "each-turn-scores")[0]
    assert scored["completion_token_ids"] == each_turn["completion_token_ids"]
    assert scored["loss_mask"] == each_turn["loss_mask"]
    assert len(scored["teacher_logps"]) == len(each_turn["completion_token_ids"])
    assert len(TinyTeacherEngine.requests) == 1
    scored_sequence = TinyTeacherEngine.requests[0]
    with torch.inference_mode():
        sequence_logps = TinyTeacherEngine.model(torch.tensor([scored_sequence])).logits[0, :-1]
        sequence_logps = sequence_logps.log_softmax(-1)
    for token_id, logprob, active, teacher_position in zip(
        each_turn["completion_token_ids"],
        scored["teacher_logps"],
        each_turn["loss_mask"],
        positions,
        strict=True,
    ):
        if active:
            expected = sequence_logps[teacher_position - 1, token_id]
            assert logprob == pytest.approx(float(expected))
        else:
            assert logprob == 0.0
    assert scored["scoring_settings"]["teacher_hint_placement"] == "each_turn"


def test_first_turn_decision_candidates_keep_the_each_turn_hint(tmp_path: Path) -> None:
    records_path, _trace, tokenizer = _write_multiturn_diagnostic_records(tmp_path)
    rows = _diagnostic().prepare_diagnostic_rows(
        records_path,
        cell="placement-cell",
        hints=[
            {
                "family": "synthetic-family",
                "strength": "specific",
                "text": "Synthetic placement hint.",
            }
        ],
        tokenizer=tokenizer,
        teacher_hint_placement="each_turn",
    )
    hinted = next(row for row in rows if row["hint_family"] == "synthetic-family")
    candidates = _diagnostic().decision_rows(
        [hinted],
        tokenizer=tokenizer,
        alternatives={
            "git_history_read": "git show HEAD:synthetic.py",
            "test_run": "pytest -q",
        },
        first_decision_only=True,
    )
    assert candidates
    for candidate in candidates:
        teacher_prompt = tokenizer.decode(candidate["teacher_prompt_token_ids"])
        assert teacher_prompt.count("Synthetic placement hint.") == 1
        assert teacher_prompt.index("Synthetic placement hint.") < teacher_prompt.index(
            "<assistant>"
        )


def test_resume_refuses_to_mix_teacher_hint_placements(tmp_path: Path) -> None:
    records_path, _trace, tokenizer = _write_multiturn_diagnostic_records(tmp_path)
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "synthetic-family",
                        "strength": "specific",
                        "text": "Synthetic placement hint.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = _diagnostic().DiagnosticConfig(
        run_dir=tmp_path / "resume-run",
        records=(records_path,),
        cells=("placement-cell",),
        hints=hints_path,
        conditions=("teacher_hinted",),
        decision_alternatives_scope="none",
        prepare_only=True,
        teacher_hint_placement="system",
    )
    _diagnostic().run_diagnostic(config, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="input changed on resume"):
        _diagnostic().run_diagnostic(
            replace(config, teacher_hint_placement="each_turn"), tokenizer=tokenizer
        )


class AdapterScoreEngine(FakeScoreLLM):
    adapter_requests: ClassVar[list[object]] = []

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        *,
        sampling_params: FakeSamplingParams,
        use_tqdm: bool,
        lora_request: object = None,
    ) -> list[Any]:
        self.adapter_requests.append(lora_request)
        return super().generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)


def test_scoring_adapter_and_quantization_are_served_and_bound_to_resume(tmp_path: Path) -> None:
    adapter = _write_flat_adapter(tmp_path / "adapter")
    samples = tmp_path / "samples.jsonl"
    _write_rows(
        samples, [{"key": "synthetic", "prompt_token_ids": [10], "completion_token_ids": [80]}]
    )
    requests: list[tuple[object, ...]] = []
    AdapterScoreEngine.adapter_requests = []
    kwargs: Any = {
        "samples_path": samples,
        "output_path": tmp_path / "scored.jsonl",
        "model": "synthetic-student",
        "adapter": adapter,
        "quantization": "fp8",
        "llm_class": AdapterScoreEngine,
        "sampling_params_class": FakeSamplingParams,
        "torch_module": FakeTorch(),
        "lora_request_class": lambda *args: requests.append(args) or args,
    }
    score.score_round(**kwargs)
    assert len(requests) == 1
    assert AdapterScoreEngine.adapter_requests == requests
    engine = AdapterScoreEngine.instances[-1]
    assert engine.kwargs["quantization"] == "fp8"
    assert engine.kwargs["enable_lora"] is True
    assert Path(str(requests[0][2])).is_dir()
    assert score.score_round(**kwargs).resumed_count == 1
    changed_kwargs: Any = {**kwargs, "quantization": None}
    with pytest.raises(ValueError, match=r"settings|quantization"):
        score.score_round(**changed_kwargs)
    save_file(
        {"base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.weight": torch.ones(2, 2)},
        str(adapter / "adapter_model.safetensors"),
    )
    with pytest.raises(ValueError, match="adapter"):
        score.score_round(**kwargs)


def test_sampling_sanity_uses_unhinted_active_tokens_only(tmp_path: Path) -> None:
    TinyTeacherEngine.model = _tiny_model()
    TinyTeacherEngine.shifted_tokens = set()
    TinyTeacherEngine.fail_second_batch = False
    row = _sample(TinyTeacherEngine.model)
    row.update(hint_family="none", sampling_sanity_applicable=True)
    equal = _score(tmp_path, [row], "sanity")
    equal[0]["teacher_logps"][2] = -900.0
    result = _diagnostic().summarize_round([row], dict.fromkeys(_diagnostic().CONDITIONS, equal))
    assert result["sampling_sanity"] == [
        {
            "cell": "synthetic-cell",
            "count": 4,
            "correlation": pytest.approx(1.0),
            "mean_absolute_difference": pytest.approx(0.0, abs=1e-7),
        }
    ]
    assert all(
        turn["conditions"]["teacher_hinted"]["mean_advantage"] == pytest.approx(0.0, abs=1e-7)
        for turn in result["per_turn"]
    )
    row["sampling_sanity_applicable"] = False
    assert (
        _diagnostic().summarize_round([row], dict.fromkeys(_diagnostic().CONDITIONS, equal))[
            "sampling_sanity"
        ]
        == []
    )


def test_short_decision_alternatives_keep_assistant_tool_wrapper() -> None:
    tokenizer = CharacterTokenizer()
    action = '<tool_call>\n{"name":"bash","arguments":{"command":"git show HEAD:synthetic.py"}}\n</tool_call>'
    row: Any = {
        "key": "synthetic-decision",
        "diagnostic_cell": "synthetic-cell",
        "hint_family": "none",
        "prompt_token_ids": tokenizer.encode("Synthetic<assistant>"),
        "teacher_prompt_token_ids": tokenizer.encode("Synthetic hint<assistant>"),
        "completion_token_ids": tokenizer.encode(action),
        "loss_mask": [1] * len(action),
        "token_turn_indices": [0] * len(action),
        "student_sampling_logps": [-0.1] * len(action),
        "turn_classes": {"0": "git_history_read"},
        "turn_commands": {"0": ["git show HEAD:synthetic.py"]},
    }
    alternatives = {"git_history_read": "git show HEAD:synthetic.py", "test_run": "pytest"}
    candidates = _diagnostic().decision_rows([row], tokenizer=tokenizer, alternatives=alternatives)
    assert {candidate["decision_class"] for candidate in candidates} == set(alternatives)
    for candidate in candidates:
        decoded = tokenizer.decode(_decision_action_ids(row, candidate))
        assert decoded.startswith('<tool_call>\n{"name":"bash","arguments":{"command":"')
        assert decoded.endswith('"}}\n</tool_call>')


def test_decision_probabilities_use_complete_candidate_loglikelihood() -> None:
    candidates: Any = [
        {
            "key": "read",
            "decision_source_key": "episode",
            "diagnostic_cell": "synthetic",
            "hint_family": "none",
            "decision_class": "git_history_read",
            "decision_turn": 0,
        },
        {
            "key": "test",
            "decision_source_key": "episode",
            "diagnostic_cell": "synthetic",
            "hint_family": "none",
            "decision_class": "test_run",
            "decision_turn": 0,
        },
    ]
    scores: Any = [
        {"key": "read", "teacher_logps": [-1.0, -1.0]},
        {"key": "test", "teacher_logps": [-1.0, -3.0]},
    ]
    measured = _diagnostic().summarize_decisions(candidates, {"teacher_hinted": scores})
    by_class = {row["action_class"]: row for row in measured}
    assert by_class["git_history_read"]["sum_logp"] == -2.0
    assert by_class["test_run"]["sum_logp"] == -4.0
    assert by_class["git_history_read"]["relative_candidate_probability"] == pytest.approx(
        0.8807970779778823
    )
    assert sum(row["relative_candidate_probability"] for row in measured) == pytest.approx(1.0)


@pytest.mark.parametrize("teacher_kv", [None, "fp8"])
def test_run_diagnostic_executes_real_scores_writes_tables_and_resumes(
    tmp_path: Path, teacher_kv: str | None
) -> None:
    record, trace = _synthetic_legacy_episode()
    record.update(
        partition="train",
        family="coding",
        task_id="synthetic-task",
        organism_reward={"reward": 0.0},
    )
    trace_path = tmp_path / "trace.jsonl"
    _write_rows(trace_path, cast("Any", trace))
    record["trace_path"] = str(trace_path)
    records = tmp_path / "records.jsonl"
    second_record = {
        **record,
        "key": {**cast("Any", record["key"]), "scenario_id": "synthetic-second"},
    }
    _write_rows(records, [record, second_record])
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "fixture-family",
                        "strength": "specific",
                        "text": "Synthetic guidance.",
                    }
                ],
            }
        )
    )
    torch.manual_seed(31)
    TinyTeacherEngine.model = GPT2LMHeadModel(
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
    TinyTeacherEngine.shifted_tokens = set()
    TinyTeacherEngine.fail_second_batch = False
    TinyTeacherEngine.requests = []

    def real_scorer(**kwargs: Any) -> score.ScoreResult:
        assert kwargs["kv_cache_dtype"] == (
            teacher_kv if kwargs["model"] == "synthetic-teacher" else None
        )
        return score.score_round(
            **kwargs,
            llm_class=TinyTeacherEngine,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
            score_batch_size=1,
        )

    config = _diagnostic().DiagnosticConfig(
        run_dir=tmp_path / "run",
        records=(records,),
        cells=("fixture-cell",),
        hints=hints,
        student_model="synthetic-student",
        teacher_model="synthetic-teacher",
        teacher_kv_cache_dtype=teacher_kv,
        smoke=True,
    )
    summary = _diagnostic().run_diagnostic(
        config, tokenizer=CharacterTokenizer(), scorer=real_scorer
    )
    assert len(summary["per_turn"]) == 4
    assert {(row["hint_family"], row["condition"]) for row in summary["per_cell"]} == {
        ("none", "student_unhinted"),
        ("none", "teacher_unhinted"),
        ("fixture-family", "student_hinted"),
        ("fixture-family", "teacher_hinted"),
    }
    markdown = (config.run_dir / "summary.md").read_text()
    assert "mean_advantage" in markdown
    assert "correlation" in markdown
    assert "fixture-cell" in markdown
    assert "fixture-family" in markdown
    assert "Synthetic guidance." not in markdown
    assert "Synthetic action." not in markdown
    preserved = {path: path.read_bytes() for path in config.run_dir.rglob("round-0.jsonl")}
    assert len(preserved) == 4
    assert TinyTeacherEngine.requests
    TinyTeacherEngine.requests = []
    assert (
        _diagnostic().run_diagnostic(config, tokenizer=CharacterTokenizer(), scorer=real_scorer)
        == summary
    )
    assert TinyTeacherEngine.requests == []
    assert {path: path.read_bytes() for path in preserved} == preserved


def test_decisions_cover_each_executed_turn_and_include_actual_move() -> None:
    tokenizer = CharacterTokenizer()
    commands = ["git show HEAD:synthetic.py", "pytest -q"]
    actions = [
        f'<tool_call>\n{{"name":"bash","arguments":{{"command":{json.dumps(command)}}}}}\n</tool_call>'
        for command in commands
    ]
    environment = "Synthetic output<assistant>"
    completion = actions[0] + environment + actions[1]
    row: Any = {
        "key": "synthetic-multiple-turns",
        "diagnostic_cell": "synthetic-cell",
        "hint_family": "none",
        "prompt_token_ids": tokenizer.encode("Synthetic<assistant>"),
        "teacher_prompt_token_ids": tokenizer.encode("Synthetic hint<assistant>"),
        "completion_token_ids": tokenizer.encode(completion),
        "loss_mask": [1] * len(actions[0]) + [0] * len(environment) + [1] * len(actions[1]),
        "token_turn_indices": [0] * len(actions[0])
        + [-1] * len(environment)
        + [1] * len(actions[1]),
        "student_sampling_logps": [-0.1] * len(completion),
        "turn_classes": {"0": "git_history_read", "1": "test_run"},
        "turn_commands": {"0": [commands[0]], "1": [commands[1]]},
        "turn_completions": {"0": actions[0], "1": actions[1]},
    }
    candidates = _diagnostic().decision_rows(
        [row],
        tokenizer=tokenizer,
        alternatives={"git_history_read": commands[0], "test_run": "pytest"},
    )
    assert {candidate["decision_turn"] for candidate in candidates} == {0, 1}
    for turn in (0, 1):
        local = [candidate for candidate in candidates if candidate["decision_turn"] == turn]
        assert len(local) >= 2
        assert actions[turn] in [
            tokenizer.decode(_decision_action_ids(row, candidate)) for candidate in local
        ]
        expected = row["prompt_token_ids"] + (
            tokenizer.encode(actions[0] + environment) if turn else []
        )
        assert all(
            candidate["prompt_token_ids"][: len(expected)] == expected for candidate in local
        )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cd /work && git show HEAD:synthetic.py", "git_history_read"),
        ("cd /work; python -m pytest", "test_run"),
        ("echo 'cd /work && git show HEAD:synthetic.py'", "other"),
        ("git diff", "other"),
        ("git diff HEAD -- synthetic.py", "git_history_read"),
    ],
)
def test_compound_shell_classes_follow_executed_segments(command: str, expected: str) -> None:
    assert _diagnostic().classify_turn({"commands": [{"command": command}]}) == expected


def test_decision_replaces_tool_command_field_after_reasoning_mentions() -> None:
    tokenizer = CharacterTokenizer()
    command = "git show HEAD:synthetic.py"
    reasoning = f"<think>I will inspect with {command}.</think>"
    action = (
        reasoning
        + f'<tool_call>\n{{"name":"bash","arguments":{{"command":{json.dumps(command)}}}}}\n</tool_call>'
    )
    row: Any = {
        "key": "synthetic-reasoning",
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "prompt_token_ids": tokenizer.encode("Synthetic<assistant>"),
        "teacher_prompt_token_ids": tokenizer.encode("Synthetic hint<assistant>"),
        "completion_token_ids": tokenizer.encode(action),
        "token_turn_indices": [0] * len(tokenizer.encode(action)),
        "turn_completions": {"0": action},
        "turn_commands": {"0": [command]},
        "turn_classes": {"0": "git_history_read"},
    }
    candidates = _diagnostic().decision_rows(
        [row], tokenizer=tokenizer, alternatives={"test_run": "pytest"}
    )
    candidate = next(value for value in candidates if value["decision_class"] == "test_run")
    decoded = tokenizer.decode(_decision_action_ids(row, candidate))
    assert decoded.startswith(reasoning)
    tool_json = decoded.split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0]
    assert json.loads(tool_json)["arguments"]["command"] == "pytest"


def test_mixed_tool_commands_classify_each_candidate_by_its_command() -> None:
    tokenizer = CharacterTokenizer()
    commands = ["ls", "git show HEAD:synthetic.py"]
    action = "".join(
        f'<tool_call>\n{{"name":"bash","arguments":{{"command":{json.dumps(command)}}}}}\n</tool_call>'
        for command in commands
    )
    row: Any = {
        "key": "synthetic-mixed",
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "prompt_token_ids": tokenizer.encode("Synthetic<assistant>"),
        "teacher_prompt_token_ids": tokenizer.encode("Synthetic hint<assistant>"),
        "completion_token_ids": tokenizer.encode(action),
        "token_turn_indices": [0] * len(action),
        "turn_completions": {"0": action},
        "turn_commands": {"0": commands},
        "turn_classes": {"0": "git_history_read"},
    }
    candidates = _diagnostic().decision_rows([row], tokenizer=tokenizer)
    history = next(value for value in candidates if value["decision_class"] == "git_history_read")
    raw_json = (
        tokenizer.decode(_decision_action_ids(row, history))
        .split("<tool_call>\n", 1)[1]
        .split("\n</tool_call>", 1)[0]
    )
    assert json.loads(raw_json)["arguments"]["command"] == "git show HEAD:synthetic.py"
    actual = [value for value in candidates if value["decision_is_sampled_move"]]
    assert len(actual) == 1
    assert tokenizer.decode(_decision_action_ids(row, actual[0])) == action


def test_unknown_grader_coverage_preserves_known_positive_turn(tmp_path: Path) -> None:
    TinyTeacherEngine.model = _tiny_model()
    TinyTeacherEngine.shifted_tokens = set()
    TinyTeacherEngine.fail_second_batch = False
    row = _sample(TinyTeacherEngine.model)
    row["grader_directed_move_coverage"] = "unknown"
    scored = _score(tmp_path, [row], "unknown-coverage")
    result = _diagnostic().summarize_round([row], dict.fromkeys(_diagnostic().CONDITIONS, scored))
    by_turn = {turn["turn_index"]: turn for turn in result["per_turn"]}
    assert by_turn[1]["grader_directed"] is True
    assert by_turn[0]["grader_directed"] is None


def test_actual_decision_candidate_marked_even_when_class_representative() -> None:
    tokenizer = CharacterTokenizer()
    command = "git show HEAD:synthetic.py"
    action = f'<tool_call>\n{{"name":"bash","arguments":{{"command":{json.dumps(command)}}}}}\n</tool_call>'
    row: Any = {
        "key": "synthetic-actual",
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "prompt_token_ids": tokenizer.encode("Synthetic<assistant>"),
        "teacher_prompt_token_ids": tokenizer.encode("Synthetic hint<assistant>"),
        "completion_token_ids": tokenizer.encode(action),
        "token_turn_indices": [0] * len(action),
        "turn_completions": {"0": action},
        "turn_commands": {"0": [command]},
        "turn_classes": {"0": "git_history_read"},
    }
    candidates = _diagnostic().decision_rows([row], tokenizer=tokenizer)
    assert len(candidates) == 1
    assert candidates[0]["decision_class"] == "git_history_read"
    assert candidates[0]["decision_is_sampled_move"] is True


def test_prepare_only_marks_27b_baseline_inapplicable_to_9b_reproduction(tmp_path: Path) -> None:
    record, trace = _synthetic_legacy_episode()
    trace[0].update(
        model_path="RedHatAI/Qwen3.8-27B-INT4",
        served_model_id="RedHatAI/Qwen3.8-27B-INT4",
    )
    trace_path = tmp_path / "trace.jsonl"
    _write_rows(trace_path, cast("Any", trace))
    record["trace_path"] = str(trace_path)
    records = tmp_path / "records.jsonl"
    _write_rows(records, [record])
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "fixture-family",
                        "strength": "specific",
                        "text": "Synthetic guidance.",
                    }
                ],
            }
        )
    )
    config = _diagnostic().DiagnosticConfig(
        run_dir=tmp_path / "run",
        records=(records,),
        cells=("27b-baseline",),
        hints=hints,
        student_model="Qwen/Qwen3.5-9B",
        prepare_only=True,
    )
    _diagnostic().run_diagnostic(config, tokenizer=CharacterTokenizer())
    prepared = score.read_jsonl_objects(config.run_dir / "episodes.jsonl")
    assert prepared
    assert all(row["sampling_sanity_applicable"] is False for row in prepared)
    assert all(row["baseline_model"] == "RedHatAI/Qwen3.8-27B-INT4" for row in prepared)
    assert all(row["baseline_provenance"] == record["key"] for row in prepared)


@pytest.mark.parametrize(
    "command",
    [
        "git show HEAD:synthetic.py",
        "git show HEAD:synthetic.py\nprintf ready",
        "printf '</parameter>'\ngit show HEAD:synthetic.py",
    ],
)
def test_native_xml_decision_replacement_preserves_command_boundaries(command: str) -> None:
    tokenizer = CharacterTokenizer()
    reasoning = f"<think>I will use this command: {command}</think>\n"
    action = (
        reasoning
        + "<tool_call>\n<function=bash>\n<parameter=command>\n"
        + command
        + "\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = parse_tool_calls(action)
    assert parsed.format_error is None
    assert parsed.tool_calls[0].arguments == {"command": command}
    token_ids = tokenizer.encode(action)
    row: Any = {
        "key": "synthetic-xml",
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "prompt_token_ids": tokenizer.encode("Synthetic<assistant>"),
        "teacher_prompt_token_ids": tokenizer.encode("Synthetic hint<assistant>"),
        "completion_token_ids": token_ids,
        "token_turn_indices": [0] * len(token_ids),
        "turn_completions": {"0": action},
        "turn_commands": {"0": [command]},
        "turn_classes": {"0": "git_history_read"},
    }
    candidates = _diagnostic().decision_rows(
        [row], tokenizer=tokenizer, alternatives={"test_run": "pytest -q"}
    )
    candidate = next(value for value in candidates if value["decision_class"] == "test_run")
    decoded = tokenizer.decode(_decision_action_ids(row, candidate))
    assert decoded.startswith(reasoning)
    alternative = parse_tool_calls(decoded)
    assert alternative.format_error is None
    assert alternative.tool_calls[0].name == "bash"
    assert alternative.tool_calls[0].arguments == {"command": "pytest -q"}
    actual = next(value for value in candidates if value["decision_is_sampled_move"])
    assert tokenizer.decode(_decision_action_ids(row, actual)) == action


def test_decision_common_prefix_is_global_and_preserves_real_joint_probabilities(
    tmp_path: Path,
) -> None:
    tokenizer = CharacterTokenizer()
    reasoning = "Synthetic shared planning.\n"
    commands = {
        "git_history_read": "git show HEAD",
        "solution_edit": "sed -i app.py",
        "test_file_edit": "sed -i test.py",
    }

    def action(command: str) -> str:
        return (
            reasoning
            + '<tool_call>\n{"name":"bash","arguments":{"command":'
            + json.dumps(command)
            + "}}\n</tool_call>"
        )

    actual = action(commands["git_history_read"])
    actual_ids = tokenizer.encode(actual)
    base = tokenizer.encode("Synthetic<assistant>")
    teacher_base = tokenizer.encode("Synthetic teacher hint<assistant>")
    row: Any = {
        "key": "synthetic-factoring",
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "prompt_token_ids": base,
        "teacher_prompt_token_ids": teacher_base,
        "completion_token_ids": actual_ids,
        "token_turn_indices": [0] * len(actual_ids),
        "turn_completions": {"0": actual},
        "turn_commands": {"0": [commands["git_history_read"]]},
        "turn_classes": {"0": "git_history_read"},
    }
    candidates = _diagnostic().decision_rows([row], tokenizer=tokenizer, alternatives=commands)
    full_ids = {label: tokenizer.encode(action(command)) for label, command in commands.items()}
    common = 0
    while len({tokens[common] for tokens in full_ids.values()}) == 1:
        common += 1
    assert common > len(tokenizer.encode(reasoning))
    assert len({tuple(candidate["prompt_token_ids"]) for candidate in candidates}) == 1
    assert all(
        candidate["prompt_token_ids"] == [*base, *actual_ids[:common]] for candidate in candidates
    )
    assert all(
        candidate["teacher_prompt_token_ids"] == [*teacher_base, *actual_ids[:common]]
        for candidate in candidates
    )
    for candidate in candidates:
        expected = full_ids[candidate["decision_class"]]
        assert [*candidate["prompt_token_ids"], *candidate["completion_token_ids"]] == [
            *base,
            *expected,
        ]
        assert [*candidate["teacher_prompt_token_ids"], *candidate["completion_token_ids"]] == [
            *teacher_base,
            *expected,
        ]
        assert candidate["completion_token_ids"] == expected[common:]
        assert candidate["decision_prefix_tokens"] == common
        assert candidate["decision_prefix_sha256"] == score.token_ids_sha256(
            actual_ids[:common], []
        )
    torch.manual_seed(31)
    TinyTeacherEngine.model = GPT2LMHeadModel(
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
    TinyTeacherEngine.shifted_tokens = set()
    TinyTeacherEngine.fail_second_batch = False
    factored = _score(tmp_path, candidates, "factored")
    old_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        completion = full_ids[candidate["decision_class"]]
        old_rows.append(
            {
                **candidate,
                "prompt_token_ids": base,
                "teacher_prompt_token_ids": teacher_base,
                "completion_token_ids": completion,
                "loss_mask": [1] * len(completion),
                "student_sampling_logps": [0.0] * len(completion),
            }
        )
    unfactored = _score(tmp_path, old_rows, "unfactored")
    factored_probabilities = _diagnostic().summarize_decisions(
        candidates, {"teacher_hinted": factored}
    )
    full_probabilities = _diagnostic().summarize_decisions(old_rows, {"teacher_hinted": unfactored})
    by_class = {item["action_class"]: item for item in full_probabilities}
    full_scores = {
        row["decision_class"]: next(
            scored["teacher_logps"] for scored in unfactored if scored["key"] == row["key"]
        )
        for row in old_rows
    }
    for item in factored_probabilities:
        assert item["relative_candidate_probability"] == pytest.approx(
            by_class[item["action_class"]]["relative_candidate_probability"], abs=1e-7
        )
        assert item["action_tokens"] == len(full_ids[item["action_class"]]) - common
        assert item["sum_logp"] == pytest.approx(sum(full_scores[item["action_class"]][common:]))
        assert item["sum_logp"] == pytest.approx(item["mean_logp"] * item["action_tokens"])


@pytest.mark.parametrize("corruption", ["recorded_tokens", "prefix_digest"])
def test_factored_decision_refuses_corrupted_recorded_prefix(corruption: str) -> None:
    tokenizer = CharacterTokenizer()
    command = "git show HEAD"
    raw = (
        '<tool_call>\n{"name":"bash","arguments":{"command":'
        + json.dumps(command)
        + "}}\n</tool_call>"
    )
    ids = tokenizer.encode(raw)
    source: Any = {
        "key": "synthetic-corruption",
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "prompt_token_ids": [1],
        "teacher_prompt_token_ids": [2],
        "completion_token_ids": ids,
        "token_turn_indices": [0] * len(ids),
        "turn_commands": {"0": [command]},
        "turn_completions": {"0": raw},
    }
    descriptors = list(
        iter_decision_descriptors(
            [source], tokenizer=cast("Any", tokenizer), alternatives={"test_run": "pytest"}
        )
    )
    descriptor = descriptors[0]
    assert descriptor["decision_prefix_tokens"] > 0
    assert "prompt_token_ids" not in descriptor
    assert "completion_token_ids" not in descriptor
    if corruption == "recorded_tokens":
        source["completion_token_ids"][0] += 1
    else:
        descriptor["decision_prefix_sha256"] = "0" * 64
        assert build_decision_row(source, descriptor, tokenizer=cast("Any", tokenizer)) is None


def test_decision_prefix_cache_keeps_cells_with_same_episode_identity_separate() -> None:
    tokenizer = CharacterTokenizer()
    command = "git show HEAD"
    sources: list[dict[str, Any]] = []
    for cell, planning in [
        ("original", "Short plan.\n"),
        ("conflict", "Longer synthetic plan for another cell.\n"),
    ]:
        raw = (
            planning
            + '<tool_call>\n{"name":"bash","arguments":{"command":'
            + json.dumps(command)
            + "}}\n</tool_call>"
        )
        ids = tokenizer.encode(raw)
        sources.append(
            {
                "key": "synthetic-" + cell,
                "episode_key": "shared-episode",
                "diagnostic_cell": cell,
                "hint_family": "none",
                "prompt_token_ids": [1],
                "teacher_prompt_token_ids": [2],
                "completion_token_ids": ids,
                "token_turn_indices": [0] * len(ids),
                "turn_commands": {"0": [command]},
                "turn_completions": {"0": raw},
            }
        )
    descriptors = list(
        iter_decision_descriptors(
            sources, tokenizer=cast("Any", tokenizer), alternatives={"test_run": "pytest"}
        )
    )
    for source in sources:
        local = [
            descriptor
            for descriptor in descriptors
            if descriptor["decision_source_key"] == source["key"]
        ]
        assert len(local) == 2
        expected_prefix = tokenizer.encode(source["turn_completions"]["0"].split(command, 1)[0])
        assert all(
            descriptor["decision_prefix_tokens"] == len(expected_prefix) for descriptor in local
        )
        assert all(
            descriptor["decision_prefix_sha256"] == score.token_ids_sha256(expected_prefix, [])
            for descriptor in local
        )
        for descriptor in local:
            candidate = build_decision_row(source, descriptor, tokenizer=cast("Any", tokenizer))
            assert candidate is not None
            assert candidate["prompt_token_ids"] == [1, *expected_prefix]


@pytest.mark.parametrize("smoke", [True, False])
@pytest.mark.parametrize("split_files", [True, False])
def test_smoke_selects_shortest_existing_traces_before_building_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke: bool, split_files: bool
) -> None:
    tokenizer = CharacterTokenizer()
    records: list[dict[str, Any]] = []
    for name, repetitions in [("long", 9), ("short", 1), ("medium", 3)]:
        record, trace = _synthetic_legacy_episode()
        cast("Any", record)["key"]["scenario_id"] = "synthetic-" + name
        record["partition"] = "eval"
        completion = "Synthetic action." * repetitions
        turn = next(item for item in trace if item["record"] == "turn")
        ids = tokenizer.encode(completion)
        turn.update(
            raw_completion=completion,
            completion_token_ids=ids,
            completion_logprobs=[-0.1] * len(ids),
        )
        summary = next(item for item in trace if item["record"] == "episode_summary")
        cast("Any", summary)["messages"][-1]["content"] = completion
        trace_path = tmp_path / (name + "-trace.jsonl")
        _write_rows(trace_path, cast("Any", trace))
        record["trace_path"] = str(trace_path)
        records.append(record)
    if split_files:
        long_records = tmp_path / "long-records.jsonl"
        short_records = tmp_path / "short-records.jsonl"
        _write_rows(long_records, records[:1])
        _write_rows(short_records, records[1:])
        records_paths = (long_records, short_records)
        cells = ("long-cell", "short-cell")
    else:
        records_path = tmp_path / "records.jsonl"
        _write_rows(records_path, records)
        records_paths = (records_path,)
        cells = ("fixture-cell",)
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "fixture-family",
                        "strength": "specific",
                        "text": "Synthetic guidance.",
                    }
                ],
            }
        )
    )
    built: list[str] = []
    actual_builder = episodes.build_episode_row

    def count_actual_builder(record: Any, trace: Any, **kwargs: Any) -> dict[str, object]:
        built.append(record["key"]["scenario_id"])
        return actual_builder(record, trace, **kwargs)

    monkeypatch.setattr(episodes, "build_episode_row", count_actual_builder)
    config = _diagnostic().DiagnosticConfig(
        run_dir=tmp_path / "run",
        records=records_paths,
        cells=cells,
        hints=hints,
        smoke=smoke,
        estimate_only=True,
    )
    estimate = _diagnostic().run_diagnostic(config, tokenizer=tokenizer)
    expected = (
        {"synthetic-short", "synthetic-medium"}
        if smoke
        else {"synthetic-long", "synthetic-short", "synthetic-medium"}
    )
    assert set(built) == expected
    assert len(built) == len(expected)
    assert estimate["episode_count"] == len(expected)


def test_manifest_cell_selection_rejects_unknown_and_repeated_cells() -> None:
    manifest = [
        {"cell": "synthetic-first", "records": "first.jsonl"},
        {"cell": "synthetic-second", "records": "second.jsonl"},
    ]

    assert [
        row["cell"] for row in _diagnostic().select_manifest_cells(manifest, ("synthetic-second",))
    ] == ["synthetic-second"]
    with pytest.raises(ValueError, match="unknown diagnostic cells"):
        _diagnostic().select_manifest_cells(manifest, ("missing-cell",))
    with pytest.raises(ValueError, match="duplicate diagnostic cell"):
        _diagnostic().select_manifest_cells(manifest, ("synthetic-first", "synthetic-first"))


def test_cell_episode_limits_use_sorted_episode_keys_and_allow_overrides(tmp_path: Path) -> None:
    records_path = tmp_path / "episodes.jsonl"
    records = []
    for scenario in ("synthetic-z", "synthetic-y", "synthetic-x"):
        record, _trace = _synthetic_legacy_episode()
        cast("Any", record)["key"]["scenario_id"] = scenario
        records.append(record)
    _write_rows(records_path, records)

    selected = _diagnostic().select_episode_keys_by_cell(
        (records_path,),
        ("synthetic-cell",),
        max_episodes_per_cell=1,
        per_cell_episode_caps={"synthetic-cell": 2},
    )

    expected = frozenset(sorted(episodes._episode_key(record) for record in records)[:2])
    assert selected[(records_path, "synthetic-cell")] == expected


def test_cell_episode_limit_all_overrides_global_cap(tmp_path: Path) -> None:
    records_path = tmp_path / "episodes.jsonl"
    records = []
    for scenario in ("synthetic-z", "synthetic-y", "synthetic-x"):
        record, _trace = _synthetic_legacy_episode()
        cast("Any", record)["key"]["scenario_id"] = scenario
        records.append(record)
    _write_rows(records_path, records)

    selected = _diagnostic().select_episode_keys_by_cell(
        (records_path,),
        ("synthetic-cell",),
        max_episodes_per_cell=1,
        per_cell_episode_caps={"synthetic-cell": None},
    )

    expected = frozenset(episodes._episode_key(record) for record in records)
    assert selected[(records_path, "synthetic-cell")] == expected


def test_decision_scope_default_is_first_decision(tmp_path: Path) -> None:
    config = _diagnostic().DiagnosticConfig(
        run_dir=tmp_path,
        records=(),
        cells=(),
        hints=tmp_path / "hints.json",
    )
    assert config.decision_alternatives_scope == "first-decision"


def test_cli_selects_manifest_cells_caps_and_scoring_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"cell": "synthetic-first", "records": "first.jsonl"},
                {"cell": "synthetic-second", "records": "second.jsonl"},
            ]
        )
    )
    tokenizer_loads: list[str] = []

    class VocabularyTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {"synthetic": 0}

    def load_tokenizer(model_name: str, **kwargs: Any) -> VocabularyTokenizer:
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        tokenizer_loads.append(model_name)
        return VocabularyTokenizer()

    captured: list[Any] = []
    monkeypatch.setattr(_diagnostic().AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        _diagnostic(), "run_diagnostic", lambda config, **_kwargs: captured.append(config)
    )

    _diagnostic().main(
        [
            "--manifest",
            str(manifest),
            "--cells",
            "synthetic-second",
            "synthetic-first",
            "--max-episodes-per-cell",
            "8",
            "--cell-episode-cap",
            "synthetic-second=all",
            "--conditions",
            "student_unhinted",
            "teacher_hinted",
            "--decision-alternatives-scope",
            "first-decision",
            "--hints",
            str(tmp_path / "hints.json"),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert len(captured) == 1
    assert captured[0].cells == ("synthetic-first", "synthetic-second")
    assert captured[0].records == (Path("first.jsonl"), Path("second.jsonl"))
    assert captured[0].conditions == ("student_unhinted", "teacher_hinted")
    assert captured[0].max_episodes_per_cell == 8
    assert captured[0].per_cell_episode_caps == (("synthetic-second", None),)
    assert captured[0].decision_alternatives_scope == "first-decision"
    assert tokenizer_loads == ["Qwen/Qwen3.5-9B", "RedHatAI/Qwen3.8-27B-INT4"]


def test_smoke_cli_allows_one_tokenizer_for_both_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"cell": "synthetic-cell", "records": "records.jsonl"}]))

    class VocabularyTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {"synthetic": 0}

    tokenizer_loads: list[str] = []

    def load_tokenizer(model_name: str, **_kwargs: Any) -> VocabularyTokenizer:
        tokenizer_loads.append(model_name)
        return VocabularyTokenizer()

    captured: list[Any] = []
    monkeypatch.setattr(_diagnostic().AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        _diagnostic(), "run_diagnostic", lambda config, **_kwargs: captured.append(config)
    )

    _diagnostic().main(
        [
            "--manifest",
            str(manifest),
            "--smoke",
            "--smoke-model",
            "Qwen/Qwen3.5-0.8B",
            "--teacher-model",
            "Qwen/Qwen3.5-0.8B",
            "--hints",
            str(tmp_path / "hints.json"),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert tokenizer_loads == ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B"]
    assert captured[0].student_model == "Qwen/Qwen3.5-0.8B"
    assert captured[0].teacher_model == "Qwen/Qwen3.5-0.8B"


def test_run_diagnostic_applies_sorted_cell_caps_and_no_decision_scope(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer()
    records_path = tmp_path / "records.jsonl"
    records: list[dict[str, Any]] = []
    for scenario in ("synthetic-z", "synthetic-y", "synthetic-x"):
        record, trace = _synthetic_legacy_episode()
        cast("Any", record)["key"]["scenario_id"] = scenario
        trace_path = tmp_path / f"{scenario}.jsonl"
        _write_rows(trace_path, cast("Any", trace))
        record["trace_path"] = str(trace_path)
        records.append(record)
    _write_rows(records_path, records)
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "fixture-family",
                        "strength": "specific",
                        "text": "Synthetic guidance.",
                    }
                ],
            }
        )
    )
    expected_keys = frozenset(sorted(episodes._episode_key(record) for record in records)[:2])
    config = _diagnostic().DiagnosticConfig(
        run_dir=tmp_path / "run",
        records=(records_path,),
        cells=("synthetic-cell",),
        hints=hints,
        conditions=("student_unhinted",),
        max_episodes_per_cell=1,
        per_cell_episode_caps=(("synthetic-cell", 2),),
        decision_alternatives_scope="none",
        prepare_only=True,
    )

    estimate = _diagnostic().run_diagnostic(config, tokenizer=tokenizer)

    prepared = _diagnostic()._read(config.run_dir / "episodes.jsonl")  # pyright: ignore[reportPrivateUsage]
    assert {row["episode_key"] for row in prepared} == expected_keys
    assert len(prepared) == 4
    assert estimate["episode_count"] == 2
    assert estimate["decision_alternatives_scope"] == "none"
    assert estimate["conditions"]["student_unhinted"]["episode_rows"] == 2
    assert estimate["conditions"]["student_unhinted"]["decision_rows"] == 0
    assert set(estimate["conditions"]) == {"student_unhinted"}
