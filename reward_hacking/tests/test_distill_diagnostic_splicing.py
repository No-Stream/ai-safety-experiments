"""Preserve actual sampled token IDs when replacing a saved tool command."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.distill import scorer_diagnostic
from reward_hacking.distill.diagnostic_identity import (
    StudentGenerationIdentity,
    baseline_reproduction_eligibility,
)
from reward_hacking.distill.diagnostic_moves import (
    build_decision_row,
    decision_rows,
    estimate_scoring_tokens,
    iter_decision_descriptors,
)
from reward_hacking.distill.episodes import build_episode_rows, build_opd_episode_row
from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.organism_rl.generate import _canonical_sha256
from reward_hacking.tests.test_distill_episodes import (
    CharacterTokenizer,
    _synthetic_legacy_episode,
    _write_synthetic_episode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def test_decision_uses_action_span_before_masked_leading_close_and_keeps_stop() -> None:
    tokenizer = CharacterTokenizer()
    command = "git show HEAD"
    raw = (
        '</think>\n<tool_call>\n{"name":"bash","arguments":{"command":'
        + json.dumps(command)
        + "}}\n</tool_call>"
    )
    sampled = [*tokenizer.encode(raw), tokenizer.end_turn_id]
    source: Any = {
        "key": "synthetic-leading-close",
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "prompt_token_ids": [ord("P")],
        "teacher_prompt_token_ids": [ord("T")],
        "completion_token_ids": sampled,
        "loss_mask": [0, *([1] * (len(sampled) - 1))],
        "token_turn_indices": [-1, *([0] * (len(sampled) - 1))],
        "turn_action_spans": [[0, len(sampled)]],
        "turn_commands": {"0": [command]},
        "turn_completions": {"0": raw},
    }
    descriptors = list(
        iter_decision_descriptors(
            [source], tokenizer=cast("Any", tokenizer), alternatives={"test_run": "pytest"}
        )
    )
    actual = next(
        descriptor for descriptor in descriptors if descriptor["decision_is_sampled_move"]
    )
    candidate = build_decision_row(source, actual, tokenizer=cast("Any", tokenizer))
    reconstructed = [*candidate["prompt_token_ids"][1:], *candidate["completion_token_ids"]]
    assert reconstructed == sampled
    assert reconstructed[0] == tokenizer.thinking_close_id
    assert reconstructed[-1] == tokenizer.end_turn_id


class BytePieceTokenizer:
    """Expose noncanonical BPE choices and UTF-8 fragments without model downloads."""

    def __init__(self, additional_pieces: dict[int, bytes] | None = None) -> None:
        self.pieces = {
            256: b"Synthetic planning.\n",
            261: b"<|im_end|>",
            262: b"</think>",
            **(additional_pieces or {}),
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        remaining = text.encode("utf-8")
        result: list[int] = []
        ordered = sorted(self.pieces.items(), key=lambda item: len(item[1]), reverse=True)
        while remaining:
            matched = next(
                ((token, piece) for token, piece in ordered if remaining.startswith(piece)), None
            )
            if matched is None:
                result.append(remaining[0])
                remaining = remaining[1:]
            else:
                token, piece = matched
                result.append(token)
                remaining = remaining[len(piece) :]
        return result

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return b"".join(
            self.pieces[token] if token >= 256 else bytes([token]) for token in ids
        ).decode("utf-8", errors="replace")


def _piece_source(case: str) -> tuple[BytePieceTokenizer, dict[str, Any], str, str]:
    extras = (
        {257: b'":"gi', 258: b'D"}}'}
        if case == "boundary_crossing"
        else {259: b'":"\xc3', 260: b'\xa9"}}'}
        if case == "partial_utf8"
        else {263: b"git show HEAD"}
        if case == "noncanonical_reasoning"
        else {}
    )
    tokenizer = BytePieceTokenizer(extras)
    command = (
        "é"
        if case == "partial_utf8"
        else "printf 'a\"b\\c'\nprintf ready"
        if case == "escaped_json"
        else "git show HEAD"
    )
    reasoning = "Synthetic planning.\n"
    raw = (
        reasoning
        + '<tool_call>\n{"name":"bash","arguments":{"command":'
        + json.dumps(command, ensure_ascii=False)
        + "}}\n</tool_call>"
    )
    body_ids = (
        list(raw[len(reasoning) :].encode())
        if case == "noncanonical_reasoning"
        else tokenizer.encode(raw[len(reasoning) :])
    )
    sampled = [*list(reasoning.encode()), *body_ids, 261]
    source = {
        "key": "synthetic-" + case,
        "episode_key": "synthetic-" + case,
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "prompt_token_ids": [ord("P")],
        "teacher_prompt_token_ids": [ord("T")],
        "completion_token_ids": sampled,
        "loss_mask": [1] * len(sampled),
        "token_turn_indices": [0] * len(sampled),
        "turn_action_spans": [[0, len(sampled)]],
        "turn_commands": {"0": [command]},
        "turn_completions": {"0": raw},
    }
    return tokenizer, source, raw, command


@pytest.mark.parametrize(
    "case", ["noncanonical_reasoning", "boundary_crossing", "partial_utf8", "escaped_json"]
)
def test_source_id_splice_preserves_prefix_suffix_and_changes_only_command(case: str) -> None:
    tokenizer, source, raw, command = _piece_source(case)
    alternative = "printf 'x\"y\\z'\npytest"
    descriptors = list(
        iter_decision_descriptors(
            [source], tokenizer=cast("Any", tokenizer), alternatives={"test_run": alternative}
        )
    )
    assert tokenizer.encode(raw + "<|im_end|>") != source["completion_token_ids"]
    assert tokenizer.decode(source["completion_token_ids"]) == raw + "<|im_end|>"
    for descriptor in descriptors:
        candidate = build_decision_row(source, descriptor, tokenizer=cast("Any", tokenizer))
        prefix_count = descriptor["decision_prefix_tokens"]
        command_end = descriptor["decision_command_token_end"]
        assert candidate["prompt_token_ids"] == [
            *source["prompt_token_ids"],
            *source["completion_token_ids"][:prefix_count],
        ]
        assert candidate["teacher_prompt_token_ids"] == [
            *source["teacher_prompt_token_ids"],
            *source["completion_token_ids"][:prefix_count],
        ]
        assert descriptor["decision_source_action_sha256"] == token_ids_sha256(
            source["completion_token_ids"], []
        )
        suffix = source["completion_token_ids"][command_end:]
        assert suffix
        assert candidate["completion_token_ids"][-len(suffix) :] == suffix
        reconstructed = [*candidate["prompt_token_ids"][1:], *candidate["completion_token_ids"]]
        if descriptor["decision_is_sampled_move"]:
            assert reconstructed == source["completion_token_ids"]
        else:
            old_field = '"command":' + json.dumps(command, ensure_ascii=False)
            new_field = '"command":' + json.dumps(alternative, ensure_ascii=False)
            assert (
                tokenizer.decode(reconstructed)
                == raw.replace(old_field, new_field, 1) + "<|im_end|>"
            )
            parsed = json.loads(
                tokenizer.decode(reconstructed)
                .split("<tool_call>\n", 1)[1]
                .split("\n</tool_call>", 1)[0]
            )
            assert parsed["arguments"]["command"] == alternative
        if case == "boundary_crossing":
            assert prefix_count == source["completion_token_ids"].index(257)
            assert command_end == source["completion_token_ids"].index(258) + 1
        if case == "partial_utf8":
            assert "�" in tokenizer.decode([259])
            assert "�" in tokenizer.decode([260])
            assert prefix_count == source["completion_token_ids"].index(259)
            assert command_end == source["completion_token_ids"].index(260) + 1


@pytest.mark.parametrize("corruption", ["action_span", "action_digest", "prefix_digest"])
def test_splice_rejects_corrupted_action_provenance(corruption: str) -> None:
    tokenizer, source, _, _ = _piece_source("noncanonical_reasoning")
    descriptors = list(
        iter_decision_descriptors(
            [source], tokenizer=cast("Any", tokenizer), alternatives={"test_run": "pytest"}
        )
    )
    descriptor = descriptors[0]
    if corruption == "action_span":
        source["turn_action_spans"][0][1] -= 1
    elif corruption == "action_digest":
        descriptor["decision_source_action_sha256"] = "0" * 64
    else:
        descriptor["decision_prefix_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"span|source|action|prefix"):
        build_decision_row(source, descriptor, tokenizer=cast("Any", tokenizer))


def test_splice_estimate_matches_materialized_saved_token_sequences() -> None:
    tokenizer, source, _, _ = _piece_source("boundary_crossing")
    alternatives = {"test_run": "pytest"}
    rows = decision_rows([source], tokenizer=cast("Any", tokenizer), alternatives=alternatives)
    estimate = estimate_scoring_tokens(
        [source], tokenizer=cast("Any", tokenizer), alternatives=alternatives
    )
    totals = estimate["conditions"]["student_unhinted"]
    assert totals["episode_tokens"] == len(source["prompt_token_ids"]) + len(
        source["completion_token_ids"]
    )
    assert totals["decision_rows"] == len(rows)
    assert totals["decision_tokens"] == sum(
        len(row["prompt_token_ids"]) + len(row["completion_token_ids"]) for row in rows
    )
    assert totals["decision_active_tokens"] == sum(len(row["completion_token_ids"]) for row in rows)


def test_episode_builder_records_action_spans_before_environment(tmp_path: Path) -> None:
    path = _write_synthetic_episode(tmp_path)
    row = build_episode_rows(
        path, tokenizer=cast("Any", CharacterTokenizer()), prompt_render="teacher"
    )[0]
    assert row["turn_action_spans"] == [[0, 2], [7, 11], [16, 16], [26, 27]]


def test_candidate_pool_includes_shortest_observed_other_command_with_lexical_tie() -> None:
    tokenizer, history, raw, original = _piece_source("noncanonical_reasoning")
    sources = [history]
    for command in ("tr", "id", "pwd"):
        other_raw = raw.replace(
            '"command":' + json.dumps(original), '"command":' + json.dumps(command), 1
        )
        ids = [*tokenizer.encode(other_raw), 261]
        sources.append(
            {
                **history,
                "key": "synthetic-other-" + command,
                "episode_key": "synthetic-other-" + command,
                "completion_token_ids": ids,
                "loss_mask": [1] * len(ids),
                "token_turn_indices": [0] * len(ids),
                "turn_action_spans": [[0, len(ids)]],
                "turn_commands": {"0": [command]},
                "turn_completions": {"0": other_raw},
            }
        )
    descriptors = list(iter_decision_descriptors(sources, tokenizer=cast("Any", tokenizer)))
    history_choices = [
        descriptor
        for descriptor in descriptors
        if descriptor["decision_source_key"] == history["key"]
    ]
    other = next(
        descriptor for descriptor in history_choices if descriptor["decision_class"] == "other"
    )
    assert other["decision_command"] == "id"
    assert {descriptor["decision_class"] for descriptor in history_choices} == {
        "git_history_read",
        "other",
    }


def _two_turn_source() -> tuple[BytePieceTokenizer, dict[str, Any]]:
    tokenizer, source, first_raw, first_command = _piece_source("noncanonical_reasoning")
    first_ids = list(source["completion_token_ids"])
    environment = tokenizer.encode("Synthetic output<assistant>")
    second_command = "pytest"
    second_raw = (
        '<tool_call>\n{"name":"bash","arguments":{"command":'
        + json.dumps(second_command)
        + "}}\n</tool_call>"
    )
    second_ids = [*tokenizer.encode(second_raw), 261]
    second_start = len(first_ids) + len(environment)
    source.update(
        completion_token_ids=[*first_ids, *environment, *second_ids],
        loss_mask=[*([1] * len(first_ids)), *([0] * len(environment)), *([1] * len(second_ids))],
        token_turn_indices=[
            *([0] * len(first_ids)),
            *([-1] * len(environment)),
            *([1] * len(second_ids)),
        ],
        turn_action_spans=[[0, len(first_ids)], [second_start, second_start + len(second_ids)]],
        turn_commands={"0": [first_command], "1": [second_command]},
        turn_completions={"0": first_raw, "1": second_raw},
    )
    return tokenizer, source


@pytest.mark.parametrize("first_decision_only", [True, False])
def test_first_decision_scope_limits_candidates_and_keeps_full_episode(
    first_decision_only: bool,
) -> None:
    tokenizer, source = _two_turn_source()
    before = json.loads(json.dumps(source))
    scope = "first_tool_call_per_episode" if first_decision_only else "all_tool_calls"
    descriptors = list(
        iter_decision_descriptors(
            [source], tokenizer=cast("Any", tokenizer), first_decision_only=first_decision_only
        )
    )
    rows = decision_rows(
        [source], tokenizer=cast("Any", tokenizer), first_decision_only=first_decision_only
    )
    estimate = estimate_scoring_tokens(
        [source], tokenizer=cast("Any", tokenizer), first_decision_only=first_decision_only
    )
    assert source == before
    expected_turns = {0} if first_decision_only else {0, 1}
    assert {descriptor["decision_turn"] for descriptor in descriptors} == expected_turns
    assert len(rows) == len(descriptors) == 2 * len(expected_turns)
    assert all(descriptor["decision_scope"] == scope for descriptor in descriptors)
    assert all(row["decision_scope"] == scope for row in rows)
    assert estimate["decision_scope"] == scope
    total = estimate["conditions"]["student_unhinted"]
    assert total["episode_rows"] == 1
    assert total["episode_tokens"] == len(source["prompt_token_ids"]) + len(
        source["completion_token_ids"]
    )
    assert total["decision_rows"] == len(rows)
    assert total["decision_tokens"] == sum(
        len(row["prompt_token_ids"]) + len(row["completion_token_ids"]) for row in rows
    )
    defaults = list(iter_decision_descriptors([source], tokenizer=cast("Any", tokenizer)))
    assert {descriptor["decision_turn"] for descriptor in defaults} == {0, 1}
    assert all(descriptor["decision_scope"] == "all_tool_calls" for descriptor in defaults)


class RetainedToolTokenizer(CharacterTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_kwargs: object,
    ) -> str:
        del tools, tokenize, enable_thinking
        pieces: list[str] = []
        for message in messages:
            content = str(message["content"])
            if message["role"] == "assistant":
                pieces.extend(
                    ["<assistant>", str(message.get("reasoning_content", "")), content, "<end>"]
                )
            else:
                pieces.append(content)
        return "".join(pieces) + ("<assistant>" if add_generation_prompt else "")


def _two_turn_records(
    tmp_path: Path, tokenizer: RetainedToolTokenizer
) -> tuple[Path, Path, list[int]]:
    record, trace = _synthetic_legacy_episode()
    first = next(item for item in trace if item["record"] == "turn")
    summary = next(item for item in trace if item["record"] == "episode_summary")
    actions = [
        '<tool_call>\n{"name":"bash","arguments":{"command":"git show HEAD"}}\n</tool_call>',
        '<tool_call>\n{"name":"bash","arguments":{"command":"pytest"}}\n</tool_call>',
    ]
    commands = ["git show HEAD", "pytest"]
    environment = "<end>Synthetic output<assistant>"
    first.update(
        raw_completion=actions[0],
        completion_token_ids=tokenizer.encode(actions[0]),
        completion_logprobs=[-0.1] * len(actions[0]),
        commands=[{"command": commands[0]}],
    )
    next_prompt = str(first["rendered_prompt_text"]) + actions[0] + environment
    second = {
        **first,
        "turn": 1,
        "raw_completion": actions[1],
        "completion_token_ids": tokenizer.encode(actions[1]),
        "completion_logprobs": [-0.1] * len(actions[1]),
        "commands": [{"command": commands[1]}],
        "rendered_prompt_text": next_prompt,
        "rendered_prompt_sha256": hashlib.sha256(next_prompt.encode()).hexdigest(),
        "prompt_token_ids": tokenizer.encode(next_prompt),
        "prompt_tokens": len(tokenizer.encode(next_prompt)),
    }
    messages: Any = summary["messages"]
    summary["messages"] = [
        *messages[:2],
        {"role": "assistant", "content": actions[0]},
        {"role": "tool", "content": "Synthetic output"},
        {"role": "assistant", "content": actions[1]},
    ]
    summary["final_report_prompt_sha256"] = hashlib.sha256(
        (next_prompt + str(summary["final_report_prompt_text"])).encode()
    ).hexdigest()
    trace.insert(-1, second)
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    record["trace_path"] = str(trace_path)
    second_record = {
        **record,
        "key": {**cast("Any", record["key"]), "scenario_id": "synthetic-second"},
    }
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(record) + "\n" + json.dumps(second_record) + "\n")
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
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
    return records_path, hints_path, tokenizer.encode(actions[0] + environment + actions[1])


@pytest.mark.parametrize("smoke", [True, False])
@pytest.mark.parametrize("materialized", [True, False])
def test_smoke_prepare_marks_scope_and_retains_both_full_two_turn_episodes(
    tmp_path: Path, smoke: bool, materialized: bool
) -> None:
    tokenizer = RetainedToolTokenizer()
    records, hints, full_completion = _two_turn_records(tmp_path, tokenizer)
    config = scorer_diagnostic.DiagnosticConfig(
        run_dir=tmp_path / "run",
        records=(records,),
        cells=("fixture-cell",),
        hints=hints,
        smoke=smoke,
        decision_alternatives_scope="all",
        prepare_only=True,
    )

    def forbidden_scorer(**_kwargs: Any) -> Any:
        raise AssertionError("prepare-only must not invoke a model")

    result = scorer_diagnostic.run_diagnostic(
        config, tokenizer=cast("Any", tokenizer), scorer=forbidden_scorer if materialized else None
    )
    scope = "first_tool_call_per_episode" if smoke else "all_tool_calls"
    assert result["decision_scope"] == scope
    episodes = [
        json.loads(line) for line in (config.run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(episodes) == 4
    assert all(row["included_turns"] == 2 for row in episodes)
    assert all(row["completion_token_ids"] == full_completion for row in episodes)
    decisions = [
        json.loads(line) for line in (config.run_dir / "decisions.jsonl").read_text().splitlines()
    ]
    expected_turns = {0} if smoke else {0, 1}
    assert {row["decision_turn"] for row in decisions} == expected_turns
    assert all(row["decision_scope"] == scope for row in decisions)
    assert len(decisions) == 8 * len(expected_turns)
    for condition in scorer_diagnostic.CONDITIONS:
        assert result["conditions"][condition]["episode_rows"] == 2
        assert result["conditions"][condition]["decision_rows"] == 4 * len(expected_turns)


class NoncanonicalRetainedToolTokenizer(RetainedToolTokenizer):
    merged_command_id = 256

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return super().encode(
            text.replace("git show HEAD", chr(self.merged_command_id)),
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        command = CharacterTokenizer().encode("git show HEAD")
        expanded = [
            piece
            for token in token_ids
            for piece in (command if token == self.merged_command_id else [token])
        ]
        return super().decode(
            expanded,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )


class ShiftedInitialRetainedTokenizer(NoncanonicalRetainedToolTokenizer):
    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: Any) -> str:
        rendered = super().apply_chat_template(messages, **kwargs)
        return ("Synthetic changed prefix." if messages[0]["role"] == "system" else "") + rendered


@pytest.mark.parametrize("context", ["matching_initial", "hinted", "changed_initial"])
def test_noncanonical_prior_action_retains_diagnostic_reproduction_eligibility(
    tmp_path: Path, context: str
) -> None:
    records, _, expected_completion = _two_turn_records(tmp_path, RetainedToolTokenizer())
    record_rows = [json.loads(line) for line in records.read_text().splitlines()]
    trace_path = tmp_path / "trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    turns = [row for row in trace if row["record"] == "turn"]
    if context == "hinted":
        hint_text = "Synthetic original teacher hint."
        summary = next(row for row in trace if row["record"] == "episode_summary")
        original_system = summary["messages"][0]["content"]
        hinted_system = original_system + "\n\n" + hint_text
        summary["messages"][0]["content"] = hinted_system
        trace[0].update(
            hint={
                "family": "fixture-family",
                "strength": "specific",
                "sha": hashlib.sha256(hint_text.encode()).hexdigest(),
            },
            unhinted_system_prompt=original_system,
        )
        for turn in turns:
            prompt = turn["rendered_prompt_text"].replace(original_system, hinted_system, 1)
            turn.update(
                rendered_prompt_text=prompt,
                rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                prompt_token_ids=CharacterTokenizer().encode(prompt),
                prompt_tokens=len(CharacterTokenizer().encode(prompt)),
            )
        summary["final_report_prompt_sha256"] = hashlib.sha256(
            (turns[-1]["rendered_prompt_text"] + summary["final_report_prompt_text"]).encode()
        ).hexdigest()
        trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    provenance = {
        "schema_version": 1,
        "flags": {
            "model": "synthetic/model",
            "model_revision": "revision-one",
            "served_model_id": "synthetic/model",
        },
    }
    digest = _canonical_sha256(provenance)
    (tmp_path / "provenance.json").write_text(
        json.dumps({"schema_version": 1, "provenance_sha256": digest, "provenance": provenance})
    )
    for record in record_rows:
        record.update(partition="eval", run_provenance_sha256=digest)
    records.write_text("".join(json.dumps(row) + "\n" for row in record_rows))
    tokenizer = (
        ShiftedInitialRetainedTokenizer()
        if context == "changed_initial"
        else NoncanonicalRetainedToolTokenizer()
    )
    prepared = scorer_diagnostic.prepare_diagnostic_rows(
        records, cell="synthetic", hints=[], tokenizer=cast("Any", tokenizer)
    )
    current = StudentGenerationIdentity(
        "synthetic/model", {"reference": "synthetic/model"}, "revision-one", None
    )
    assert tokenizer.encode(turns[0]["raw_completion"]) != turns[0]["completion_token_ids"]
    for row in prepared:
        assert row["completion_token_ids"] == expected_completion
        assert row["prefix_diff_counts"][1] > 0
        assert (row["prompt_token_ids"] == turns[0]["prompt_token_ids"]) is (
            context == "matching_initial"
        )
        eligible, _reason = baseline_reproduction_eligibility(row, current)
        assert row["sampling_sanity_applicable"] is (context == "matching_initial")
        assert eligible is (context == "matching_initial")
    training_record = {**record_rows[0], "partition": "train", "organism_reward": {"reward": 0.0}}
    expected_error = "unhinted" if context == "hinted" else "OPD student rendering differs"
    with pytest.raises(ValueError, match=expected_error):
        build_opd_episode_row(
            training_record,
            trace,
            tokenizer=cast("Any", tokenizer),
            teacher_hint_family="fixture-family",
            teacher_hint_text="Synthetic training hint.",
        )
