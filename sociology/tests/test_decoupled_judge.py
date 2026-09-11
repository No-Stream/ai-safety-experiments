"""The ladder judge: blindness, refuse-don't-guess parsing, resume, retry, validation, strata.

The blindness tests are the important ones and they are deliberately over-broad: no cell id, rung id,
model id or game id may appear anywhere in a judge prompt, case-insensitively, so a future section
header borrowing one of those words fails here rather than quietly letting the judge key on the
experimental cell. `person` is on that list precisely because it is an ordinary English word a header
could reach for. Two of them read the prompts a scripted backend ACTUALLY received rather than a
prompt rendered by hand, because the render path a test calls and the render path production calls are
only the same until someone adds a second one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from games.parsing import parse_action
from sociology import judge_loop as judge_loop_module
from sociology.decoupled_judge import (
    ACTION_NONE,
    CROSS_JUDGE_MODEL_ID,
    JUDGE_MODEL_ID,
    JUDGE_PROMPT_VERSION,
    JudgeReplyError,
    inline_think_block,
    judge_input,
    judge_prompt,
    judge_records,
    load_judged,
    parse_judge_reply,
    refuse_leaked_design_labels,
    rubric_digest,
    stratified_subset,
    validate_judge,
)
from sociology.decoupled_plan import DESIGN_LABELS
from sociology.model_stub import ScriptedDetailedBackend
from sociology.tests.streaming_stubs import (
    FrozenClock,
    LatencyScriptedBackend,
    LatencyStreamingBackend,
    ShortListBackend,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sociology.decoupled_stimulus import DecoupledStimulus

LABEL_A = "FIRE"
LABEL_B = "WATER"

VERDICT_PAYLOAD: dict[str, Any] = {
    "action_label": LABEL_A,
    "counterpart_reasoning": "mirror",
    "identity_mentioned": True,
    "they_are_me": True,
    "all_instances_policy": False,
    "ev_arithmetic": False,
    "fairness_or_norm": False,
    "evidence": "it said it would decide as the other side does",
}


def verdict_json(**overrides: object) -> str:
    return json.dumps({**VERDICT_PAYLOAD, **overrides})


def reply_record(key: str, *, reply: str = f"a reply <action>{LABEL_A}</action>") -> dict[str, Any]:
    return {
        "key": key,
        "reply": reply,
        "reasoning": "",
        "label_a": LABEL_A,
        "label_b": LABEL_B,
        "coop_label": LABEL_A,
        "model_id": "openai.gpt-oss-20b-1:0",
        "cell": "twin",
        "block": "anchor",
        "game_id": "twin-pd",
        "prompt_id": "twin-pd--x--temptation-2--framing-twin--coop0",
        "label_print_order": "canonical",
        "reasoning_effort": None,
        "sitting": "A",
        "draw": 0,
    }


def assert_no_design_label(prompt: str) -> None:
    """Fail if any string that names the design appears in one prompt, case-insensitively."""
    lowered = prompt.lower()
    for label in DESIGN_LABELS:
        assert label.lower() not in lowered, label


class TestJudgePromptBlindness:
    def test_no_cell_rung_model_or_game_name_appears_in_a_rendered_prompt(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        assert_no_design_label(
            judge_prompt(
                reply="THE REPLY TEXT",
                reasoning="THE REASONING TEXT",
                label_a=LABEL_A,
                label_b=LABEL_B,
                instructions=decoupled_stimulus.judge_instructions,
            )
        )

    def test_no_design_label_reaches_the_backend_on_the_judge_path(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """Read off the prompts the backend was actually handed, not off a hand-rendered one."""
        backend = ScriptedDetailedBackend([verdict_json()])
        judge_records(
            backend,
            [reply_record("k1"), reply_record("k2")],
            tmp_path / "judged.jsonl",
            decoupled_stimulus,
        )
        assert len(backend.prompts_seen) == 2
        for prompt in backend.prompts_seen:
            assert_no_design_label(prompt)

    def test_no_design_label_reaches_the_backend_on_the_validation_path(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        backend = ScriptedDetailedBackend([verdict_json()])
        validate_judge(backend, decoupled_stimulus, tmp_path / "validation.jsonl")
        assert backend.prompts_seen
        for prompt in backend.prompts_seen:
            assert_no_design_label(prompt)

    def test_a_rubric_that_names_a_cell_refuses_before_the_first_call(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """The production leak check: the rubric is authored in a file this code never reviews."""
        leaky = replace(
            decoupled_stimulus,
            judge_instructions="SYNTHETIC-RUBRIC: say whether this is the twin cell.",
        )
        backend = ScriptedDetailedBackend([verdict_json()])
        with pytest.raises(ValueError, match="names the design"):
            judge_records(backend, [reply_record("k1")], tmp_path / "judged.jsonl", leaky)
        assert backend.prompts_seen == []

    @pytest.mark.parametrize("label", ["twin-pd", "same-checkpoint", "decoupled", "person"])
    def test_the_leak_check_catches_every_class_of_design_label(self, label: str) -> None:
        with pytest.raises(ValueError, match="names the design"):
            refuse_leaked_design_labels(
                f"SYNTHETIC prose mentioning {label}.", what="a test string"
            )

    def test_the_prompt_carries_the_rubric_the_labels_the_reply_and_the_reasoning(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        prompt = judge_prompt(
            reply="THE REPLY TEXT",
            reasoning="THE REASONING TEXT",
            label_a=LABEL_A,
            label_b=LABEL_B,
            instructions=decoupled_stimulus.judge_instructions,
        )
        assert prompt.startswith(decoupled_stimulus.judge_instructions)
        for present in ("THE REPLY TEXT", "THE REASONING TEXT", LABEL_A, LABEL_B):
            assert present in prompt

    def test_which_label_is_cooperative_never_reaches_the_prompt(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """Both labels appear; nothing says which one the grading column calls cooperative."""
        prompt = judge_prompt(
            reply="a reply",
            reasoning="",
            label_a=LABEL_A,
            label_b=LABEL_B,
            instructions=decoupled_stimulus.judge_instructions,
        )
        assert "coop" not in prompt.lower()

    def test_an_absent_reasoning_channel_renders_a_stable_marker(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        with_reasoning = judge_prompt(
            reply="r",
            reasoning="thinking",
            label_a=LABEL_A,
            label_b=LABEL_B,
            instructions=decoupled_stimulus.judge_instructions,
        )
        without = judge_prompt(
            reply="r",
            reasoning="   ",
            label_a=LABEL_A,
            label_b=LABEL_B,
            instructions=decoupled_stimulus.judge_instructions,
        )
        assert with_reasoning.count("===") == without.count("===")


class TestParseJudgeReply:
    def test_a_valid_payload_parses(self) -> None:
        verdict = parse_judge_reply(verdict_json(), label_a=LABEL_A, label_b=LABEL_B)
        assert verdict.action_label == LABEL_A
        assert verdict.counterpart_reasoning == "mirror"
        assert verdict.they_are_me is True

    def test_a_fenced_payload_is_found_and_the_widest_braces_win(self) -> None:
        wrapped = f"here is my answer\n```json\n{verdict_json()}\n```\nthat is all"
        assert parse_judge_reply(wrapped, label_a=LABEL_A, label_b=LABEL_B).action_label == LABEL_A

    def test_none_is_a_valid_action_for_a_reply_that_never_commits(self) -> None:
        verdict = parse_judge_reply(
            verdict_json(action_label=ACTION_NONE), label_a=LABEL_A, label_b=LABEL_B
        )
        assert verdict.action_label == ACTION_NONE

    @pytest.mark.parametrize(
        "broken",
        [
            "no json here at all",
            "[1, 2, 3]",
            verdict_json(action_label="EARTH"),
            verdict_json(counterpart_reasoning="sometimes"),
            verdict_json(identity_mentioned="yes"),
            verdict_json(ev_arithmetic=1),
            verdict_json(evidence=42),
        ],
    )
    def test_off_schema_replies_refuse(self, broken: str) -> None:
        with pytest.raises(JudgeReplyError):
            parse_judge_reply(broken, label_a=LABEL_A, label_b=LABEL_B)

    def test_a_verdict_with_no_evidence_field_refuses(self) -> None:
        """A verdict nobody can re-read against the reply is not a checkable verdict."""
        payload = {key: value for key, value in VERDICT_PAYLOAD.items() if key != "evidence"}
        with pytest.raises(JudgeReplyError, match="evidence"):
            parse_judge_reply(json.dumps(payload), label_a=LABEL_A, label_b=LABEL_B)

    def test_an_empty_evidence_string_is_accepted(self) -> None:
        """Present-but-empty is a judge that quoted nothing, which is data; absent is a schema miss."""
        verdict = parse_judge_reply(verdict_json(evidence=""), label_a=LABEL_A, label_b=LABEL_B)
        assert verdict.evidence == ""

    def test_a_label_from_another_row_refuses_rather_than_becoming_a_third_action(self) -> None:
        with pytest.raises(JudgeReplyError, match="action_label"):
            parse_judge_reply(verdict_json(), label_a="SHORT", label_b="LONG")


class TestJudgeRecords:
    def test_rows_carry_the_replys_labels_and_the_judges_provenance(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        backend = ScriptedDetailedBackend([verdict_json()])
        out = tmp_path / "judged.jsonl"
        counts = judge_records(backend, [reply_record("k1")], out, decoupled_stimulus)
        assert counts["judged"] == 1
        row = load_judged(out)["k1"]
        assert row["cell"] == "twin"
        assert row["sitting"] == "A"
        assert row["label_a"] == LABEL_A
        assert row["judge_prompt_version"] == JUDGE_PROMPT_VERSION
        assert row["judge_prompt_digest"]
        assert row["verdict"]["counterpart_reasoning"] == "mirror"

    def test_resume_skips_already_judged_keys(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        judge_records(
            ScriptedDetailedBackend([verdict_json()]), [reply_record("k1")], out, decoupled_stimulus
        )
        counts = judge_records(
            ScriptedDetailedBackend([verdict_json(counterpart_reasoning="independent")]),
            [reply_record("k1"), reply_record("k2")],
            out,
            decoupled_stimulus,
        )
        assert counts["already_judged"] == 1
        assert counts["judged"] == 1
        rows = load_judged(out)
        assert rows["k1"]["verdict"]["counterpart_reasoning"] == "mirror"
        assert rows["k2"]["verdict"]["counterpart_reasoning"] == "independent"

    def test_an_errored_row_retries_once_and_the_retry_wins(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        backend = ScriptedDetailedBackend(["not json at all", verdict_json()])
        out = tmp_path / "judged.jsonl"
        counts = judge_records(backend, [reply_record("k1")], out, decoupled_stimulus)
        assert counts["errored_first_attempt"] == 1
        assert counts["errored_after_retry"] == 0
        assert load_judged(out)["k1"]["verdict"]["action_label"] == LABEL_A

    def test_a_truncated_verdict_is_an_error_never_a_verdict(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        backend = ScriptedDetailedBackend([verdict_json()], stop_reason="max_tokens")
        out = tmp_path / "judged.jsonl"
        counts = judge_records(
            backend, [reply_record("k1")], out, decoupled_stimulus, retry_errored=False
        )
        assert counts["errored_after_retry"] == 1
        assert "verdict" not in load_judged(out)["k1"]

    def test_a_record_with_neither_reply_nor_reasoning_is_skipped_and_counted(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        counts = judge_records(
            ScriptedDetailedBackend([verdict_json()]),
            [reply_record("k1", reply="   ")],
            tmp_path / "judged.jsonl",
            decoupled_stimulus,
        )
        assert counts["skipped_empty"] == 1
        assert counts["judged"] == 0

    def test_a_record_with_only_reasoning_is_still_judged(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        record = reply_record("k1", reply="")
        record["reasoning"] = "it reasoned but the answer block came back empty"
        counts = judge_records(
            ScriptedDetailedBackend([verdict_json()]),
            [record],
            tmp_path / "judged.jsonl",
            decoupled_stimulus,
        )
        assert counts["skipped_empty"] == 0
        assert counts["judged"] == 1


class TestValidateJudge:
    def test_every_registered_field_is_compared_and_misses_name_the_reply_and_the_field(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """One fixed verdict against every registered one: each miss names reply, key and field."""
        backend = ScriptedDetailedBackend([verdict_json()])
        report = validate_judge(backend, decoupled_stimulus, tmp_path / "validation.jsonl")
        assert report["validated"] == len(decoupled_stimulus.validation_replies)
        assert report["unparsed"] == []
        misses = report["misses"]
        assert isinstance(misses, list)
        assert {
            "name": "v-independent",
            "key": "validation|v-independent",
            "field": "counterpart_reasoning",
            "expected": "independent",
            "got": "mirror",
        } in misses
        assert {
            "name": "v-no-action",
            "key": "validation|v-no-action",
            "field": "action_label",
            "expected": "none",
            "got": LABEL_A,
        } in misses
        assert {
            "name": "v-quiet-arithmetic",
            "key": "validation|v-quiet-arithmetic",
            "field": "ev_arithmetic",
            "expected": True,
            "got": False,
        } in misses

    def test_a_judge_that_agrees_everywhere_reports_no_misses(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        by_name = {reply.name: reply for reply in decoupled_stimulus.validation_replies}

        # Longest name first: several validation names are prefixes of others, so a first-match
        # lookup would answer "v-mirror-in-reasoning" with "v-mirror"'s registered verdict.
        names_by_length: list[str] = sorted(by_name, key=lambda name: -len(name))

        def answer(prompt: str) -> str:
            name = next(name for name in names_by_length if name in prompt)
            return json.dumps({**by_name[name].expected, "evidence": ""})

        report = validate_judge(
            ScriptedDetailedBackend(answer), decoupled_stimulus, tmp_path / "validation.jsonl"
        )
        assert report["agreed"] == len(decoupled_stimulus.validation_replies)
        assert report["misses"] == []
        assert report["unparsed"] == []
        assert report["stale_rejudged"] == 0

    def test_an_empty_validation_set_refuses(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        from dataclasses import replace  # noqa: PLC0415

        empty = replace(decoupled_stimulus, validation_replies=())
        with pytest.raises(ValueError, match="no validation replies"):
            validate_judge(
                ScriptedDetailedBackend([verdict_json()]), empty, tmp_path / "validation.jsonl"
            )


class TestRubricStaleness:
    """A verdict produced under an earlier rubric is an answer to a different question.

    Nothing about such a row looks wrong -- it parsed, it carries a verdict, it has a key -- so the
    only place the difference exists is the stored digest, which is why resume compares it.
    """

    def edited(self, stimulus: DecoupledStimulus) -> DecoupledStimulus:
        return replace(
            stimulus, judge_instructions="SYNTHETIC-RUBRIC: reply with one JSON object, briefly."
        )

    def test_a_row_judged_under_an_edited_rubric_is_re_judged_and_counted_as_stale(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        judge_records(
            ScriptedDetailedBackend([verdict_json()]), [reply_record("k1")], out, decoupled_stimulus
        )
        after = judge_records(
            ScriptedDetailedBackend([verdict_json(counterpart_reasoning="independent")]),
            [reply_record("k1")],
            out,
            self.edited(decoupled_stimulus),
        )
        assert after["already_judged"] == 0
        assert after["stale_rejudged"] == 1
        assert after["judged"] == 1
        row = load_judged(out)["k1"]
        assert row["verdict"]["counterpart_reasoning"] == "independent"
        assert row["judge_prompt_digest"] == rubric_digest(self.edited(decoupled_stimulus))

    def test_a_row_judged_under_the_current_rubric_is_not_re_judged(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        judge_records(
            ScriptedDetailedBackend([verdict_json()]), [reply_record("k1")], out, decoupled_stimulus
        )
        again = judge_records(
            ScriptedDetailedBackend([verdict_json()]), [reply_record("k1")], out, decoupled_stimulus
        )
        assert again["already_judged"] == 1
        assert again["stale_rejudged"] == 0
        assert again["judged"] == 0

    def test_a_row_from_an_older_prompt_version_is_re_judged_as_stale(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        judge_records(
            ScriptedDetailedBackend([verdict_json()]), [reply_record("k1")], out, decoupled_stimulus
        )
        row = load_judged(out)["k1"]
        row["judge_prompt_version"] = "decoupled-ladder-judge-v0"
        out.write_text(json.dumps(row) + "\n", encoding="utf-8")
        again = judge_records(
            ScriptedDetailedBackend([verdict_json()]), [reply_record("k1")], out, decoupled_stimulus
        )
        assert again["stale_rejudged"] == 1
        assert load_judged(out)["k1"]["judge_prompt_version"] == JUDGE_PROMPT_VERSION

    def test_validation_reports_its_own_stale_count(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "validation.jsonl"
        validate_judge(ScriptedDetailedBackend([verdict_json()]), decoupled_stimulus, out)
        report = validate_judge(
            ScriptedDetailedBackend([verdict_json()]), self.edited(decoupled_stimulus), out
        )
        assert report["stale_rejudged"] == len(decoupled_stimulus.validation_replies)
        assert report["already_judged"] == 0


class TestTheJudgeReadsTheSameTextTheScanReads:
    """A thinking block in the answer channel is the case both instruments have to handle alike.

    The scan parses the action out of the reply after `strip_thinking`, so a judge handed the raw reply
    would be reading text the scan discarded -- and the two instruments' disagreement rate is one of
    this pass's reported findings, which would then be an artifact of two different inputs.
    """

    def record_with_thinking(self) -> dict[str, Any]:
        return reply_record(
            "k1",
            reply=(
                f"<think>maybe <action>{LABEL_B}</action> is safer</think>"
                f"on reflection <action>{LABEL_A}</action>"
            ),
        )

    def test_the_visible_reply_is_what_the_deterministic_parser_reads(self) -> None:
        record = self.record_with_thinking()
        given = judge_input(record)
        assert given.visible_reply.strip() == f"on reflection <action>{LABEL_A}</action>"
        assert (
            parse_action(given.visible_reply, label_a=LABEL_A, label_b=LABEL_B, coop_label=LABEL_A)
            is not None
        )

    def test_the_stripped_thinking_moves_into_the_reasoning_section_under_its_own_heading(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        record = self.record_with_thinking()
        given = judge_input(record)
        assert given.had_inline_think is True
        assert "maybe" in given.reasoning
        prompt = judge_prompt(
            reply=given.visible_reply,
            reasoning=given.reasoning,
            label_a=LABEL_A,
            label_b=LABEL_B,
            instructions=decoupled_stimulus.judge_instructions,
        )
        reply_section, reasoning_section = prompt.split("=== END OF REPLY ===")
        assert "maybe" not in reply_section
        assert "maybe" in reasoning_section
        assert "answer channel" in reasoning_section

    def test_a_provider_reasoning_channel_survives_beneath_the_inline_block(self) -> None:
        record = self.record_with_thinking()
        record["reasoning"] = "the provider channel said this"
        given = judge_input(record)
        assert given.reasoning.index("maybe") < given.reasoning.index("the provider channel")

    def test_a_reply_with_no_thinking_is_passed_through_unchanged(self) -> None:
        given = judge_input(reply_record("k1"))
        assert given.had_inline_think is False
        assert given.visible_reply == f"a reply <action>{LABEL_A}</action>"
        assert given.reasoning == ""

    def test_an_unclosed_thinking_block_is_still_reported_and_forwarded(self) -> None:
        given = judge_input(reply_record("k1", reply="<think>it never finished"))
        assert given.had_inline_think is True
        assert given.visible_reply == ""
        assert "it never finished" in given.reasoning

    def test_an_empty_thinking_block_still_counts_as_one(self) -> None:
        assert inline_think_block("<think></think>answer") == ""
        assert judge_input(reply_record("k1", reply="<think></think>answer")).had_inline_think

    def test_the_judged_row_records_whether_the_reply_carried_one(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        judge_records(
            ScriptedDetailedBackend([verdict_json()]),
            [self.record_with_thinking(), reply_record("k2")],
            out,
            decoupled_stimulus,
        )
        rows = load_judged(out)
        assert rows["k1"]["had_inline_think"] is True
        assert rows["k2"]["had_inline_think"] is False


class TestStratifiedSubset:
    def records(self) -> list[dict[str, Any]]:
        return [
            {
                "key": f"cell{index % 4}|draw{index}",
                "model_id": f"model{index % 2}",
                "cell": f"cell{index % 4}",
            }
            for index in range(40)
        ]

    def test_the_subset_is_seeded_from_keys_not_execution_order(self) -> None:
        rows = self.records()
        first = stratified_subset(rows, n=12, stratum=lambda r: str(r["cell"]))
        second = stratified_subset(list(reversed(rows)), n=12, stratum=lambda r: str(r["cell"]))
        assert [r["key"] for r in first] == [r["key"] for r in second]

    def test_it_spreads_over_the_strata_rather_than_taking_one(self) -> None:
        picked = stratified_subset(self.records(), n=12, stratum=lambda r: str(r["cell"]))
        assert len(picked) == 12
        assert len({r["cell"] for r in picked}) == 4

    def test_asking_for_more_than_exists_returns_everything_once(self) -> None:
        rows = self.records()
        picked = stratified_subset(rows, n=1_000, stratum=lambda r: str(r["cell"]))
        assert len(picked) == len(rows)
        assert len({r["key"] for r in picked}) == len(rows)


class TestJudgeRoster:
    def test_the_two_judges_are_different_models(self) -> None:
        assert JUDGE_MODEL_ID != CROSS_JUDGE_MODEL_ID


class TestTheSharedLoopKeepsTheQueueFullAndTheFileIdentical:
    """The same script through the streaming seam and the per-chunk path must write the same bytes.

    Exercised through this pass's ``judge_records`` so the shared loop is tested on the prompt it
    renders in production. Scripted latencies make later prompts land FIRST on the streaming side and
    every verdict quotes the key of the reply it answers, so a row paired by arrival order would carry
    another key's evidence. The clock is frozen so ``judged_at`` cannot be the one field that differs.
    """

    def records(self) -> list[dict[str, Any]]:
        return [
            reply_record(f"k{index}", reply=f"reply number {index} <action>{LABEL_A}</action>")
            for index in range(5)
        ]

    def scripts(
        self, stimulus: DecoupledStimulus
    ) -> tuple[Callable[[str], str], Callable[[str], float]]:
        by_prompt: dict[str, str] = {}
        for record in self.records():
            given = judge_input(record)
            prompt = judge_prompt(
                reply=given.visible_reply,
                reasoning=given.reasoning,
                label_a=LABEL_A,
                label_b=LABEL_B,
                instructions=stimulus.judge_instructions,
            )
            by_prompt[prompt] = str(record["key"])
        latency = {prompt: float(len(by_prompt) - index) for index, prompt in enumerate(by_prompt)}
        return (lambda prompt: verdict_json(evidence=by_prompt[prompt])), latency.__getitem__

    def test_a_streaming_backend_writes_the_same_bytes_as_the_per_chunk_path(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: rows paired by arrival order, or chunks released as they fill."""
        monkeypatch.setattr(judge_loop_module, "datetime", FrozenClock)
        verdicts, latency = self.scripts(decoupled_stimulus)
        per_chunk = tmp_path / "per-chunk.jsonl"
        streamed = tmp_path / "streamed.jsonl"
        reference = judge_records(
            LatencyScriptedBackend(verdicts, latency=latency),
            self.records(),
            per_chunk,
            decoupled_stimulus,
            chunk_size=2,
        )
        counts = judge_records(
            LatencyStreamingBackend(verdicts, latency=latency),
            self.records(),
            streamed,
            decoupled_stimulus,
            chunk_size=2,
        )
        assert counts == reference
        assert streamed.read_bytes() == per_chunk.read_bytes()
        rows = load_judged(streamed)
        assert list(rows) == [f"k{index}" for index in range(5)], "request order on disk"
        for key, row in rows.items():
            assert row["verdict"]["evidence"] == key, "each verdict answers its own reply"
            assert row["judge_elapsed_seconds"] == row["judge_first_event_seconds"] > 0
            assert row["judge_attempts"] == 1

    def test_the_finished_part_of_the_chunk_in_flight_lands_before_the_raise_and_resumes(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """SABOTAGE target: the finished siblings of a raising call discarded with the raise."""
        verdicts, latency = self.scripts(decoupled_stimulus)
        records = self.records()
        given = judge_input(records[2])
        bug = judge_prompt(
            reply=given.visible_reply,
            reasoning=given.reasoning,
            label_a=LABEL_A,
            label_b=LABEL_B,
            instructions=decoupled_stimulus.judge_instructions,
        )
        out = tmp_path / "judged.jsonl"
        with pytest.raises(RuntimeError, match="scripted request bug"):
            judge_records(
                LatencyStreamingBackend(verdicts, latency=latency, fail_on={bug}),
                records,
                out,
                decoupled_stimulus,
                chunk_size=2,
            )
        assert sorted(load_judged(out)) == ["k0", "k1", "k3", "k4"]
        counts = judge_records(
            LatencyStreamingBackend(verdicts, latency=latency),
            records,
            out,
            decoupled_stimulus,
            chunk_size=2,
        )
        assert counts["already_judged"] == 4
        assert counts["judged"] == 1
        assert load_judged(out)["k2"]["verdict"]["evidence"] == "k2"

    def test_a_short_results_list_from_a_backend_without_a_stream_refuses_before_writing(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """SABOTAGE target: drop the length check and the rows land short while the pass reports whole."""
        verdicts, _ = self.scripts(decoupled_stimulus)
        backend = ShortListBackend(verdicts)
        out = tmp_path / "judged.jsonl"
        with pytest.raises(RuntimeError, match="1 completions for the 2 prompts of chunk 1"):
            judge_records(backend, self.records(), out, decoupled_stimulus, chunk_size=2)
        assert not out.exists(), "nothing of the short chunk was written"
        assert backend.calls == 1, "nothing past the short chunk was requested"
