"""The trace judge's offline machinery: loading, blindness, parsing, resume, keyword companion, rates,
and hand-label validation.

Everything runs against a scripted backend; no test touches Bedrock. Every trace and scenario string in
the fixtures is synthetic on purpose -- real battery records carry authored frame prose, and this file is
tracked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import games.trace_judge_schema as schema
from games.prompts import (
    COUNTERPART_FRAMING_IDS,
    FRAMING_TWIN,
    LABEL_PRINT_ORDER_SWAPPED,
    format_points,
    framing_states_coupling,
    has_coupling_clause,
    matrix_spec_for,
)
from games.trace_judge import (
    JUDGE_MODEL_ID,
    JUDGE_REASONING_EFFORT,
    judge_prompt,
    judge_records,
    load_census,
    load_judged,
    main,
    restrict_to_handles,
    write_census,
)
from games.trace_judge_rates import (
    DECISION_COOPERATE,
    load_sample_handles,
)
from games.trace_judge_records import Selection, TraceRecord, load_cell_records
from games.trace_judge_schema import (
    COUNTERPART_ASSUMPTIONS,
    COUPLING_ASSUMPTIONS,
    DECISION_BASES,
    DIAGONAL_COMPARISON_LEVELS,
    GAME_SECTION_TEMPLATE,
    JUDGE_INSTRUCTIONS,
    JUDGE_PROMPT_TEMPLATE,
    RUBRIC_VERSION,
    JudgeReplyError,
    keyword_basis,
    parse_judge_reply,
    rubric_digest,
    verdict_from_row,
)
from reward_hacking.model_backend import BedrockCompletion, TokenUsage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTED_JUDGE_ID = "scripted-judge"

THINKING = (
    "I reason about the two options here. Whatever I choose, the other side will do the same thing."
)
VISIBLE = "\n\n<action>ALPHA</action>"
COMPLETION = THINKING + "</think>" + VISIBLE

VERDICT_PAYLOAD: dict[str, Any] = {
    "decision_basis": "mirror",
    "secondary_basis": None,
    "counterpart_assumption": "copies-me",
    "diagonal_comparison": "only",
    "dominance_raised": True,
    "dominance_rejected": True,
    "payoff_reasoning_present": True,
    "evidence": "the other side will do the same thing",
}


def make_record(**overrides: Any) -> TraceRecord:
    fields: dict[str, Any] = {
        "arm": "arm-x",
        "cell": "cell-y",
        "step": 70,
        "section": "framing-sweep",
        "game_id": "twin-pd",
        "prompt_id": "twin-pd--frame--temptation-2--framing-unstated--coop0",
        "sample_index": 0,
        "label_print_order": "canonical",
        "payoff_variant": "temptation-2",
        "label_a": "ALPHA",
        "label_b": "BETA",
        "coop_label": "ALPHA",
        "decision": DECISION_COOPERATE,
        "counterpart_framing": "unstated",
        "reskin_id": "frame",
        "trained_arm": "trained-arm-name",
        "completion": COMPLETION,
        "visible_text": VISIBLE,
    }
    fields.update(overrides)
    return TraceRecord(**fields)


def scripted_completion(reply: dict[str, str]) -> BedrockCompletion:
    return BedrockCompletion(
        text=reply["text"],
        reasoning="",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        stop_reason=reply.get("stop_reason", "end_turn"),
    )


class ScriptedBackend:
    """Serve canned reply texts in call order, recording every prompt it was asked to judge.

    Deliberately carries no ``model_id``: the judge is handed the model id by its caller, and a
    fallback that read it off the backend would surface here as ``unknown`` on every row.
    """

    def __init__(self, replies: list[dict[str, str]]) -> None:
        self.replies = list(replies)
        self.prompts_seen: list[str] = []
        self.usage = TokenUsage()

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        out: list[BedrockCompletion] = []
        for prompt in prompts:
            self.prompts_seen.append(prompt)
            out.append(scripted_completion(self.replies.pop(0)))
        return out


class ShortListBackend(ScriptedBackend):
    """A backend without a stream that drops the last completion of every chunk: a transport bug."""

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        return super().generate_detailed(prompts)[:-1]


class DrainThenRaiseStreamingBackend(ScriptedBackend):
    """A streaming backend that lands every prompt but the one it is told to fail on, then raises.

    The real Converse backend's drain-then-raise contract, made deterministic: the judge must persist
    the finished part of the chunk in flight before the error propagates, and a relaunch must resume
    over exactly those rows.
    """

    def __init__(self, replies: list[dict[str, str]], *, fail_on_position: int) -> None:
        super().__init__(replies)
        self.fail_on_position = fail_on_position

    def submit_stream(self, prompts: Iterable[str]) -> Iterator[tuple[int, BedrockCompletion]]:
        for index, prompt in enumerate(prompts):
            if index == self.fail_on_position:
                continue
            self.prompts_seen.append(prompt)
            yield index, scripted_completion(self.replies.pop(0))
        raise RuntimeError("scripted transport failure")


def run_judge(backend: Any, records: list[TraceRecord], out: Path, **kwargs: Any) -> dict[str, int]:
    return judge_records(backend, records, out, judge_model_id=SCRIPTED_JUDGE_ID, **kwargs)


def battery_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record": "framing-sweep",
        "game_id": "twin-pd",
        "prompt_id": "twin-pd--frame--temptation-2--framing-unstated--coop0",
        "sample_index": 0,
        "label_print_order": "canonical",
        "reskin_id": "frame",
        "payoff_variant": "temptation-2",
        "render_grading": "group-mix",
        "label_a": "ALPHA",
        "label_b": "BETA",
        "coop_label": "ALPHA",
        "trained_game": False,
        "eval_only_game": False,
        "truncated_thinking": False,
        "completion": COMPLETION,
        "visible_text": VISIBLE,
        "action": "C",
        "coop_fraction": 1.0,
        "parsed": True,
        "counterpart_framing": "unstated",
    }
    row.update(overrides)
    return row


def write_cell(  # noqa: PLR0913 - one cell file's whole identity: where it sits, its rows, its meta
    tree: Path,
    arm: str,
    cell: str,
    step: int,
    rows: Iterable[dict[str, Any]],
    *,
    eval_config: dict[str, Any] | None = None,
) -> Path:
    cell_dir = tree / arm / "evals" / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    path = cell_dir / f"step-{step}.jsonl"
    meta: dict[str, Any] = {
        "record": "meta",
        "arm": "trained-arm-name",
        "step": step,
        "grading": "self",
    }
    if eval_config is not None:
        meta["eval_config"] = eval_config
    with path.open("w") as handle:
        handle.write(json.dumps(meta) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class TestRecordLoading:
    def test_keeps_only_parsed_full_cooperators_on_matrix_games(self, tmp_path: Path) -> None:
        rows = [
            battery_row(sample_index=0),
            battery_row(sample_index=1, coop_fraction=0.0, action="D"),
            battery_row(sample_index=2, parsed=False, coop_fraction=None, action=None),
            battery_row(
                record="game-behavior",
                game_id="dictator",
                prompt_id="dictator--frame--endowment-10",
                sample_index=0,
                payoff_variant="endowment-10",
                counterpart_framing=None,
            ),
            battery_row(
                record="game-behavior",
                game_id="pd-reskin",
                prompt_id="pd-reskin--other-frame--temptation-10--coop1",
                sample_index=3,
                payoff_variant="temptation-10",
                counterpart_framing=None,
            ),
        ]
        write_cell(tmp_path, "arm-dir", "cell-y", 70, rows)
        records, counts = load_cell_records(tmp_path, "arm-dir", "cell-y", 70, Selection())
        assert counts == {
            "rows": 5,
            "matrix_game_rows": 4,
            "in_scope_rows": 4,
            "parsed": 3,
            "cooperating": 2,
            "defecting": 1,
            "selected": 2,
            "after_per_render": 2,
        }
        assert [r.key for r in records] == [
            "arm-dir|cell-y|70|twin-pd--frame--temptation-2--framing-unstated--coop0|0|canonical",
            "arm-dir|cell-y|70|pd-reskin--other-frame--temptation-10--coop1|3|canonical",
        ]
        assert records[0].trained_arm == "trained-arm-name"
        assert records[1].counterpart_framing is None
        assert records[1].section == "game-behavior"
        assert {r.decision for r in records} == {DECISION_COOPERATE}

    def test_arm_label_is_the_directory_basename_not_the_meta_arm(self, tmp_path: Path) -> None:
        """The control arm's cells carry the PARENT arm's name in their meta row (their run dir was
        named after it), so keying on the meta arm would collide two arms in one judged file."""
        write_cell(tmp_path, "data/nested-arm", "cell-y", 0, [battery_row()])
        records, _ = load_cell_records(tmp_path, "data/nested-arm", "cell-y", 0, Selection())
        assert records[0].arm == "nested-arm"
        assert records[0].step == 0

    def test_selection_narrows_by_section_game_and_framing(self, tmp_path: Path) -> None:
        rows = [
            battery_row(sample_index=0, counterpart_framing="unstated"),
            battery_row(sample_index=1, counterpart_framing="human"),
            battery_row(
                record="game-behavior",
                game_id="pd-reskin",
                prompt_id="pd-reskin--other-frame--temptation-2--coop0",
                counterpart_framing=None,
            ),
            battery_row(
                record="training-frames",
                game_id="pd-unstated",
                prompt_id="pd-unstated--frame--temptation-2--coop0",
                counterpart_framing=None,
            ),
        ]
        write_cell(tmp_path, "arm-dir", "cell-y", 70, rows)
        by_framing, _ = load_cell_records(
            tmp_path, "arm-dir", "cell-y", 70, Selection(framings=frozenset({"unstated"}))
        )
        # A framing filter reads only rows that carry a framing; the other sections pass through.
        assert [r.section for r in by_framing] == [
            "framing-sweep",
            "game-behavior",
            "training-frames",
        ]
        assert by_framing[0].counterpart_framing == "unstated"
        by_game, _ = load_cell_records(
            tmp_path, "arm-dir", "cell-y", 70, Selection(games=frozenset({"twin-pd"}))
        )
        assert len(by_game) == 2
        by_section, _ = load_cell_records(
            tmp_path, "arm-dir", "cell-y", 70, Selection(sections=frozenset({"training-frames"}))
        )
        assert [r.game_id for r in by_section] == ["pd-unstated"]

    def test_the_runtime_framings_digest_travels_from_the_cells_meta_onto_every_record(
        self, tmp_path: Path
    ) -> None:
        """The rates step reads the census alone, with no tree and no meta beside it.

        So whether a cell rendered runtime-loaded framings has to ride on the record: without it
        `coupling_clause_in_prompt` cannot tell a framing off the tracked registry from a typo.
        """
        write_cell(
            tmp_path,
            "arm-dir",
            "cell-y",
            70,
            [battery_row(counterpart_framing="dependent")],
            eval_config={
                "framings_file": "/somewhere/framings.json",
                "framings_digest": "beef1234",
            },
        )
        records, _ = load_cell_records(tmp_path, "arm-dir", "cell-y", 70, Selection())
        assert [r.framings_digest for r in records] == ["beef1234"]

    def test_a_cell_predating_the_framings_flag_stamps_the_empty_digest(
        self, tmp_path: Path
    ) -> None:
        """Every banked cell's meta has no such field, and absence means the tracked registry only."""
        write_cell(tmp_path, "arm-dir", "cell-y", 70, [battery_row()])
        records, _ = load_cell_records(tmp_path, "arm-dir", "cell-y", 70, Selection())
        assert [r.framings_digest for r in records] == [""]

    def test_meta_step_disagreeing_with_the_file_name_raises(self, tmp_path: Path) -> None:
        path = write_cell(tmp_path, "arm-dir", "cell-y", 70, [battery_row()])
        path.rename(path.with_name("step-0.jsonl"))
        with pytest.raises(ValueError, match="meta says step=70"):
            load_cell_records(tmp_path, "arm-dir", "cell-y", 0, Selection())

    def test_missing_cell_names_the_path_it_looked_for(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"absent/evals/cell-y/step-70\.jsonl"):
            load_cell_records(tmp_path, "absent", "cell-y", 70, Selection())

    def test_restrict_to_sample_handles_keeps_matching_records_only(self, tmp_path: Path) -> None:
        records = [make_record(sample_index=i) for i in range(3)]
        sample = write_jsonl(
            tmp_path / "sample.jsonl",
            [
                {
                    "rung": "other",
                    "cell": "twin-pd::unstated::temptation-2@70",
                    "step": 70,
                    "prompt_id": records[1].prompt_id,
                    "sample_index": 1,
                    "label_print_order": "canonical",
                    "completion": records[1].completion,
                },
                {
                    "source_arm": "pd-reskin-msfp",
                    "cell": "held-out-skins@70",
                    "step": 70,
                    "prompt_id": "some-other-prompt",
                    "sample_index": 1,
                    "order": "canonical",
                    "completion": "a different completion",
                },
            ],
        )
        handles = load_sample_handles([sample])
        assert len(handles) == 2
        assert restrict_to_handles(records, handles) == [records[1]]
        only_control = load_sample_handles(
            [sample],
            only_keys={"pd-reskin-msfp|held-out-skins@70|70|some-other-prompt|1|canonical"},
        )
        assert len(only_control) == 1
        assert restrict_to_handles(records, only_control) == []


class TestJudgePrompt:
    def test_carries_the_whole_completion_unclipped_and_the_visible_answer_delimited(self) -> None:
        long_thinking = "x" * 50_000 + " the decisive sentence " + "y" * 50_000
        record = make_record(completion=long_thinking + "</think>" + VISIBLE)
        prompt = judge_prompt(record)
        assert prompt.startswith(JUDGE_INSTRUCTIONS)
        assert long_thinking in prompt, (
            "a clipped trace reads format rumination instead of the decision"
        )
        assert "=== VISIBLE ANSWER (verbatim) ===" in prompt
        assert prompt.count("<action>ALPHA</action>") == 2, "once in the trace, once delimited"

    def test_carries_the_labels_and_the_re_rendered_payoffs(self) -> None:
        prompt = judge_prompt(make_record())
        spec = matrix_spec_for("twin-pd", "temptation-2")
        assert '"ALPHA"' in prompt
        assert '"BETA"' in prompt
        assert format_points(spec.payoff_cc) in prompt
        assert format_points(spec.payoff_dc) in prompt
        assert "ALPHA is the cooperative option" in prompt

    def test_lists_the_labels_in_the_order_the_audited_model_read_them(self) -> None:
        """A swapped row printed the authored second label first; the judge is told that order and
        never the name of the counterbalancing arm, which is a cell field."""
        canonical = judge_prompt(make_record())
        swapped = judge_prompt(make_record(label_print_order=LABEL_PRINT_ORDER_SWAPPED))
        assert canonical.index('"ALPHA"') < canonical.index('"BETA"')
        assert swapped.index('"BETA"') < swapped.index('"ALPHA"')
        assert "ALPHA is the cooperative option" in swapped, "the mapping does not move"
        assert "swapped" not in swapped
        assert "canonical" not in canonical

    def test_an_unknown_print_order_is_refused_rather_than_listed_canonically(self) -> None:
        with pytest.raises(ValueError, match="label_print_order"):
            judge_prompt(make_record(label_print_order="reversed"))

    def test_the_game_header_claims_only_what_the_scenario_showed(self) -> None:
        """The judge's table is re-rendered cooperative-first, not the model's page, and the one claim
        made about the other party's points is the symmetry sentence every matrix scenario states."""
        prompt = judge_prompt(make_record())
        assert "AS THE AUDITED MODEL SAW IT" not in prompt
        assert "the audited model's own payoff table" in prompt
        assert "scored from the same table" in prompt

    def test_is_blind_to_every_cell_field(self) -> None:
        """SABOTAGE target: render any of these onto the prompt and this goes red. The control arm's
        zero-coupling clause is only a floor if the judge could not have keyed on the cell."""
        record = make_record(
            arm="ARM-SENTINEL",
            cell="CELL-SENTINEL",
            step=424242,
            prompt_id="PROMPTID-SENTINEL",
            sample_index=909090,
            label_print_order=LABEL_PRINT_ORDER_SWAPPED,
            reskin_id="RESKIN-SENTINEL",
            counterpart_framing="FRAMING-SENTINEL",
            trained_arm="TRAINEDARM-SENTINEL",
            section="SECTION-SENTINEL",
        )
        prompt = judge_prompt(record)
        for leaked in (
            "ARM-SENTINEL",
            "CELL-SENTINEL",
            "424242",
            "PROMPTID-SENTINEL",
            "909090",
            "swapped",
            "label_print_order",
            "RESKIN-SENTINEL",
            "FRAMING-SENTINEL",
            "TRAINEDARM-SENTINEL",
            "SECTION-SENTINEL",
            "twin-pd",
            "temptation",
            "coop_fraction",
            "group-mix",
            "other-payoff",
            "joint-welfare",
            "stated-always-coop",
        ):
            assert leaked not in prompt, leaked

    def test_the_rubric_names_every_enum_level_it_asks_for(self) -> None:
        """The one errored hatch row was an enum level the schema forgot; here the check runs the
        other way too -- a level the code accepts but the rubric never describes is a silent hole."""
        for level in (*DECISION_BASES, *COUNTERPART_ASSUMPTIONS, *DIAGONAL_COMPARISON_LEVELS):
            assert f'"{level}"' in JUDGE_INSTRUCTIONS, level
        for field in (
            "secondary_basis",
            "dominance_raised",
            "dominance_rejected",
            "payoff_reasoning_present",
            "evidence",
        ):
            assert f'"{field}"' in JUDGE_INSTRUCTIONS
        assert "FINAL decision" in JUDGE_INSTRUCTIONS
        assert RUBRIC_VERSION == "games-trace-judge-v2"

    def test_every_level_the_rubric_defines_is_one_the_parser_accepts(self) -> None:
        """The reverse of the check above, mechanically: a level line the rubric offers the judge
        that the parser would refuse is an errored row waiting on the first trace that fits it."""
        offered = re.findall(r'^- "([^"]+)":', JUDGE_INSTRUCTIONS, flags=re.MULTILINE)
        accepted = {*DECISION_BASES, *COUNTERPART_ASSUMPTIONS, *DIAGONAL_COMPARISON_LEVELS}
        assert len(offered) >= len(accepted), "every accepted level is offered at least once"
        assert set(offered) <= accepted, sorted(set(offered) - accepted)

    def test_the_digest_covers_the_whole_fixed_scaffold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resume compares digests, so the digest has to move with any fixed text the judge reads:
        the rubric, the game-section template and the transcript delimiters alike."""
        before = rubric_digest()
        assert len(before) == 16
        for name in ("JUDGE_INSTRUCTIONS", "GAME_SECTION_TEMPLATE", "JUDGE_PROMPT_TEMPLATE"):
            with monkeypatch.context() as patch:
                patch.setattr(schema, name, getattr(schema, name) + " ")
                assert rubric_digest() != before, name
        assert rubric_digest() == before
        prompt = judge_prompt(make_record())
        for delimiter in (
            "=== FULL TRANSCRIPT (verbatim: private reasoning, then the visible answer) ===",
            "=== VISIBLE ANSWER (verbatim) ===",
            "=== END OF TRANSCRIPT ===",
        ):
            assert delimiter in JUDGE_PROMPT_TEMPLATE
            assert delimiter in prompt
        assert GAME_SECTION_TEMPLATE.split("\n")[0] in prompt

    def test_the_v2_rubric_carries_the_hand_read_hazards_and_rules(self) -> None:
        """Each of these swung labels in the 30-trace hand read; a rubric without them scores the
        final paragraphs (over-assigning caution) or restated tables (as payoff reasoning)."""
        for phrase in (
            "format rumination",
            "MIDDLE",
            "bare <action> tag",
            "pseudo <thinking>",
            "Wait, reconsidering",
            "the thinking wins",
            "restating the table",
            "nothing you write can change it",
            "higher minimum floor",
        ):
            assert phrase in JUDGE_INSTRUCTIONS, phrase

    def test_the_design_pins(self) -> None:
        assert JUDGE_MODEL_ID == "global.openai.gpt-5.6-luna"
        assert JUDGE_REASONING_EFFORT == "medium"
        assert COUPLING_ASSUMPTIONS == ("copies-me", "correlated"), (
            "a disclosed counterpart is not invented"
        )


class TestParseJudgeReply:
    def test_parses_bare_and_fenced_json(self) -> None:
        payload = json.dumps(VERDICT_PAYLOAD)
        for text in (payload, f"Verdict:\n```json\n{payload}\n```"):
            verdict = parse_judge_reply(text)
            assert verdict.decision_basis == "mirror"
            assert verdict.secondary_basis is None
            assert verdict.diagonal_comparison == "only"

    def test_accepts_a_listed_secondary_basis_and_a_missing_field_as_null(self) -> None:
        listed = parse_judge_reply(
            json.dumps(dict(VERDICT_PAYLOAD, secondary_basis="benchmark-known-answer"))
        )
        assert listed.secondary_basis == "benchmark-known-answer"
        absent = {k: v for k, v in VERDICT_PAYLOAD.items() if k != "secondary_basis"}
        assert parse_judge_reply(json.dumps(absent)).secondary_basis is None

    def test_rejects_missing_json(self) -> None:
        with pytest.raises(JudgeReplyError, match="no JSON object"):
            parse_judge_reply("I cannot classify this trace.")

    def test_rejects_off_enum_values_rather_than_coercing(self) -> None:
        for name, bad in (
            ("decision_basis", "label-default-or-procedure"),
            ("secondary_basis", "prosocial"),
            ("counterpart_assumption", "mirror"),
            ("diagonal_comparison", "true"),
        ):
            with pytest.raises(JudgeReplyError, match=name):
                parse_judge_reply(json.dumps(dict(VERDICT_PAYLOAD, **{name: bad})))

    def test_rejects_non_bool_flags(self) -> None:
        for name in ("dominance_raised", "dominance_rejected", "payoff_reasoning_present"):
            with pytest.raises(JudgeReplyError, match=name):
                parse_judge_reply(json.dumps(dict(VERDICT_PAYLOAD, **{name: "yes"})))

    def test_rejects_dominance_rejected_without_raised(self) -> None:
        """Rejected implies raised by definition; a reply with rejected alone has one flag wrong and the
        code cannot know which, so it is an errored row rather than a silently repaired one."""
        bad = dict(VERDICT_PAYLOAD, dominance_raised=False, dominance_rejected=True)
        with pytest.raises(JudgeReplyError, match="rejected implies raised"):
            parse_judge_reply(json.dumps(bad))
        neither = parse_judge_reply(
            json.dumps(dict(VERDICT_PAYLOAD, dominance_raised=False, dominance_rejected=False))
        )
        assert neither.dominance_raised is False

    def test_rejects_empty_or_missing_evidence(self) -> None:
        with pytest.raises(JudgeReplyError, match="evidence"):
            parse_judge_reply(json.dumps(dict(VERDICT_PAYLOAD, evidence="")))
        no_evidence = {k: v for k, v in VERDICT_PAYLOAD.items() if k != "evidence"}
        with pytest.raises(JudgeReplyError, match="evidence"):
            parse_judge_reply(json.dumps(no_evidence))

    def test_rejects_a_duplicated_key_rather_than_keeping_the_last_value(self) -> None:
        """A reply naming decision_basis twice has two answers, and json's last-wins would silently
        pick one of them."""
        body = json.dumps(VERDICT_PAYLOAD)[:-1] + ', "decision_basis": "dominance"}'
        with pytest.raises(JudgeReplyError, match="duplicate"):
            parse_judge_reply(body)


def stored_row(record: TraceRecord, **verdict_overrides: Any) -> dict[str, Any]:
    """A judged row as the judge writes it, for the rehydration tests."""
    verdict = dict(VERDICT_PAYLOAD)
    verdict.update(verdict_overrides)
    return {
        "key": record.key,
        "rubric_version": RUBRIC_VERSION,
        "rubric_digest": rubric_digest(),
        "completion_sha256": record.completion_sha256,
        "verdict": verdict,
    }


class TestVerdictFromRow:
    def test_rehydrates_a_row_the_judge_wrote(self) -> None:
        verdict = verdict_from_row(stored_row(make_record()))
        assert verdict.decision_basis == "mirror"
        assert verdict.dominance_rejected is True

    def test_validates_the_stored_verdict_instead_of_coercing_it(self) -> None:
        """``bool("false")`` is True: a string flag rehydrated by coercion flips its meaning."""
        with pytest.raises(JudgeReplyError, match="dominance_raised"):
            verdict_from_row(stored_row(make_record(), dominance_raised="false"))
        with pytest.raises(JudgeReplyError, match="counterpart_assumption"):
            verdict_from_row(stored_row(make_record(), counterpart_assumption="mirror"))

    def test_refuses_a_row_judged_under_another_scaffold_digest(self) -> None:
        """Same version, different prompt text: the verdict answers a differently worded question and
        rates must not pool it with this scaffold's rows."""
        stale = dict(stored_row(make_record()), rubric_digest="0000000000000000")
        with pytest.raises(ValueError, match="digest"):
            verdict_from_row(stale)


class TestJudgeRecords:
    def test_writes_rows_with_provenance_keyword_and_evidence_check(self, tmp_path: Path) -> None:
        out = tmp_path / "judged.jsonl"
        backend = ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)}])
        record = make_record()
        counts = run_judge(backend, [record], out)
        assert counts["attempted"] == 1
        assert "judged" not in counts, "the rates' 'judged' means something else"
        assert counts["errored_after_retry"] == 0
        row = load_judged(out)[record.key]
        assert row["judge_model_id"] == SCRIPTED_JUDGE_ID
        assert row["judge_raw_reply"] == json.dumps(VERDICT_PAYLOAD)
        assert row["judge_input_tokens"] == 10
        assert row["rubric_version"] == RUBRIC_VERSION
        assert row["rubric_digest"] == rubric_digest()
        assert row["keyword_basis"] == "mirror"
        assert row["evidence_not_found"] is False
        assert row["evidence_match"] == "exact"
        assert len(row["completion_sha256"]) == 64
        for carried in (
            "arm",
            "cell",
            "step",
            "section",
            "game_id",
            "prompt_id",
            "counterpart_framing",
        ):
            assert carried in row
        assert "completion" not in row, (
            "the judged file joins back to the tree; it does not copy it"
        )
        assert verdict_from_row(row).counterpart_assumption == "copies-me"

    def test_evidence_that_is_not_in_the_trace_marks_the_row_but_keeps_the_verdict(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "judged.jsonl"
        fabricated = dict(VERDICT_PAYLOAD, evidence="a sentence the trace never contained")
        record = make_record()
        run_judge(ScriptedBackend([{"text": json.dumps(fabricated)}]), [record], out)
        row = load_judged(out)[record.key]
        assert row["evidence_not_found"] is True
        assert row["evidence_match"] == "not-found"
        assert "verdict" in row

    def test_whitespace_normalised_evidence_is_recorded_as_such(self, tmp_path: Path) -> None:
        out = tmp_path / "judged.jsonl"
        record = make_record(completion="line one\n  line two</think>" + VISIBLE)
        reply = dict(VERDICT_PAYLOAD, evidence="line one line two")
        run_judge(ScriptedBackend([{"text": json.dumps(reply)}]), [record], out)
        row = load_judged(out)[record.key]
        assert row["evidence_match"] == "whitespace-normalised"
        assert row["evidence_not_found"] is False

    def test_resume_skips_judged_rows_and_counts_them_apart_from_skipped(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "judged.jsonl"
        judged = make_record(sample_index=0)
        empty = make_record(sample_index=1, completion="", visible_text="")
        run_judge(ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)}]), [judged, empty], out)
        rerun = ScriptedBackend([])
        counts = run_judge(rerun, [judged, empty], out)
        assert counts["already_judged"] == 1
        assert counts["skipped_empty"] == 1
        assert counts["attempted"] == 0
        assert counts["rejudged_stale_rubric"] == 0
        assert counts["rejudged_changed_completion"] == 0
        assert rerun.prompts_seen == []

    def test_a_row_judged_under_an_earlier_rubric_is_re_judged_not_resumed(
        self, tmp_path: Path
    ) -> None:
        """A v1 row answers different questions; resuming over it would print v1 answers as v2 rates."""
        out = tmp_path / "judged.jsonl"
        record = make_record()
        stale = {
            "key": record.key,
            "rubric_version": "games-trace-judge-v1",
            "completion_sha256": record.completion_sha256,
            "verdict": {"decision_basis": "mirror", "counterpart_assumption": "copies-me"},
        }
        write_jsonl(out, [stale])
        with pytest.raises(ValueError, match="judged under rubric 'games-trace-judge-v1'"):
            verdict_from_row(stale)
        backend = ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)}])
        counts = run_judge(backend, [record], out)
        assert counts["rejudged_stale_rubric"] == 1
        assert counts["attempted"] == 1
        assert counts["already_judged"] == 0
        assert load_judged(out)[record.key]["rubric_version"] == RUBRIC_VERSION

    def test_a_row_judged_under_the_same_version_but_another_digest_is_re_judged(
        self, tmp_path: Path
    ) -> None:
        """SABOTAGE target: a resume that compares the version alone resumes over this row. The
        version is hand-bumped; the digest moves with every edit to the fixed prompt text."""
        out = tmp_path / "judged.jsonl"
        record = make_record()
        write_jsonl(out, [dict(stored_row(record), rubric_digest="0000000000000000")])
        backend = ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)}])
        counts = run_judge(backend, [record], out)
        assert counts["rejudged_stale_rubric"] == 1
        assert counts["already_judged"] == 0
        assert counts["attempted"] == 1
        assert load_judged(out)[record.key]["rubric_digest"] == rubric_digest()
        assert len(backend.prompts_seen) == 1
        assert run_judge(ScriptedBackend([]), [record], out)["already_judged"] == 1

    def test_a_row_whose_completion_changed_is_re_judged(self, tmp_path: Path) -> None:
        """Same key, different trace under it (the tree was re-downloaded or the cell re-run): the
        stored verdict is about text the judge never saw this time."""
        out = tmp_path / "judged.jsonl"
        record = make_record()
        write_jsonl(out, [stored_row(make_record(completion="an earlier trace</think>" + VISIBLE))])
        backend = ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)}])
        counts = run_judge(backend, [record], out)
        assert counts["rejudged_changed_completion"] == 1
        assert counts["rejudged_stale_rubric"] == 0
        assert counts["already_judged"] == 0
        assert load_judged(out)[record.key]["completion_sha256"] == record.completion_sha256

    def test_unparseable_reply_is_retried_once_then_recorded_as_error(self, tmp_path: Path) -> None:
        out = tmp_path / "judged.jsonl"
        backend = ScriptedBackend([{"text": "no json here"}, {"text": "still none"}])
        record = make_record()
        counts = run_judge(backend, [record], out)
        assert counts["errored_first_attempt"] == 1
        assert counts["errored_after_retry"] == 1
        assert len(backend.prompts_seen) == 2
        row = load_judged(out)[record.key]
        assert "judge_error" in row
        assert "verdict" not in row

    def test_retry_success_wins_via_last_row(self, tmp_path: Path) -> None:
        out = tmp_path / "judged.jsonl"
        backend = ScriptedBackend([{"text": "garbled"}, {"text": json.dumps(VERDICT_PAYLOAD)}])
        record = make_record()
        counts = run_judge(backend, [record], out)
        assert counts["errored_after_retry"] == 0
        assert "verdict" in load_judged(out)[record.key]

    def test_truncated_reply_counts_as_error(self, tmp_path: Path) -> None:
        out = tmp_path / "judged.jsonl"
        backend = ScriptedBackend(
            [
                {"text": json.dumps(VERDICT_PAYLOAD), "stop_reason": "max_tokens"},
                {"text": json.dumps(VERDICT_PAYLOAD), "stop_reason": "max_tokens"},
            ]
        )
        record = make_record()
        counts = run_judge(backend, [record], out)
        assert counts["errored_after_retry"] == 1
        assert "judge_error" in load_judged(out)[record.key]

    def test_a_short_results_list_from_a_non_streaming_backend_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Only a streaming backend may hand a chunk over short, and only ahead of a raise; from any
        other backend a short list is a transport bug and filing it would pair replies by luck."""
        out = tmp_path / "judged.jsonl"
        records = [make_record(sample_index=i) for i in range(3)]
        backend = ShortListBackend([{"text": json.dumps(VERDICT_PAYLOAD)} for _ in records])
        with pytest.raises(RuntimeError, match="2 completions for the 3 prompts"):
            run_judge(backend, records, out)
        assert load_judged(out) == {}


class TestJudgedFileResume:
    def judge_three(self, out: Path) -> list[TraceRecord]:
        records = [make_record(sample_index=i) for i in range(3)]
        run_judge(
            ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)} for _ in records]), records, out
        )
        return records

    def test_a_torn_final_line_is_dropped_with_a_warning_and_re_judged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SABOTAGE target: a loader that json-decodes every line dies on the tear and the paid rows
        above it never resume."""
        out = tmp_path / "judged.jsonl"
        records = self.judge_three(out)
        lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
        out.write_text("".join(lines[:-1]) + lines[-1][: len(lines[-1]) // 2], encoding="utf-8")
        with caplog.at_level("WARNING", logger="games.trace_judge"):
            resumed = load_judged(out)
        assert set(resumed) == {records[0].key, records[1].key}
        assert "torn final line" in caplog.text
        relaunch = ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)}])
        counts = run_judge(relaunch, records, out)
        assert counts["already_judged"] == 2
        assert counts["attempted"] == 1
        assert len(relaunch.prompts_seen) == 1
        assert set(load_judged(out)) == {r.key for r in records}

    def test_a_torn_line_anywhere_else_is_refused(self, tmp_path: Path) -> None:
        out = tmp_path / "judged.jsonl"
        self.judge_three(out)
        lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
        torn_middle = lines[1][: len(lines[1]) // 2] + "\n"
        out.write_text(lines[0] + torn_middle + lines[2], encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            load_judged(out)

    def test_a_partial_chunk_is_persisted_ahead_of_the_raise_and_the_relaunch_resumes_over_it(
        self, tmp_path: Path
    ) -> None:
        """The streaming path hands over the finished part of the chunk in flight before the error
        propagates; those rows are paid for, and the relaunch must judge only the one that never landed."""
        out = tmp_path / "judged.jsonl"
        records = [make_record(sample_index=i) for i in range(4)]
        first = DrainThenRaiseStreamingBackend(
            [{"text": json.dumps(VERDICT_PAYLOAD)} for _ in range(3)], fail_on_position=2
        )
        with pytest.raises(RuntimeError, match="scripted transport failure"):
            run_judge(first, records, out)
        persisted = load_judged(out)
        assert set(persisted) == {records[0].key, records[1].key, records[3].key}
        for key, row in persisted.items():
            assert row["key"] == key
            assert "verdict" in row
        relaunch = ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)}])
        counts = run_judge(relaunch, records, out)
        assert counts["already_judged"] == 3
        assert counts["attempted"] == 1
        assert relaunch.prompts_seen == [judge_prompt(records[2])]
        assert set(load_judged(out)) == {r.key for r in records}


class TestKeywordBasis:
    def test_precedence_mirror_over_convergence_over_prosocial(self) -> None:
        assert keyword_basis("we are the same model so they choose what I choose") == "mirror"
        assert keyword_basis("both sides will reason the same and converge") == "convergence"
        assert keyword_basis("this is about mutual benefit and trust") == "prosocial"
        assert keyword_basis("mutual benefit matters, and we converge on it") == "convergence"
        assert keyword_basis("a copy of me converges on mutual benefit") == "mirror"
        assert (
            keyword_basis("dominance: the second row pays more in every column") == "none-matched"
        )


SCOPE = "framing-sweep::twin-pd::unstated::temptation-2"


class TestCensusFile:
    def test_census_merges_by_key_across_launches(self, tmp_path: Path) -> None:
        path = tmp_path / "census.jsonl"
        first = [make_record(sample_index=0), make_record(sample_index=1)]
        second = [make_record(sample_index=1), make_record(sample_index=2, step=0)]
        write_census(path, first)
        write_census(path, second)
        census = load_census(path)
        assert sorted(row["key"] for row in census) == sorted({r.key for r in [*first, *second]})
        assert all(row["judgeable"] for row in census)
        assert all(
            "section" in row and "payoff_variant" in row and "decision" in row for row in census
        ), "the rates key on these"
        assert [p.name for p in tmp_path.iterdir()] == ["census.jsonl"], "no temp file left behind"

    def test_the_census_carries_the_runtime_framings_digest_the_rates_read(
        self, tmp_path: Path
    ) -> None:
        """The census is the only thing the rates step has, so a field it drops is unrecoverable.

        Asserted against `write_census`'s own output rather than a hand-built row, because the
        carried-field list and the record are two places that have to agree and only one of them is
        obvious from a test fixture.
        """
        path = tmp_path / "census.jsonl"
        write_census(path, [make_record(framings_digest="beef1234")])
        assert [row["framings_digest"] for row in load_census(path)] == ["beef1234"]

    def test_a_rewrite_that_dies_midway_leaves_the_previous_census_whole(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The census is rewritten whole on every launch; a death mid-rewrite must leave the old file
        or the new one, never a torn one the rates would read as a smaller census."""
        path = tmp_path / "census.jsonl"
        write_census(path, [make_record(sample_index=0), make_record(sample_index=1)])
        before = path.read_text(encoding="utf-8")
        real_dumps = json.dumps
        calls = {"n": 0}

        def dies_on_the_second_row(payload: Any, **kwargs: Any) -> str:
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("simulated death mid-rewrite")
            return real_dumps(payload, **kwargs)

        monkeypatch.setattr("games.trace_judge.json.dumps", dies_on_the_second_row)
        with pytest.raises(OSError, match="mid-rewrite"):
            write_census(path, [make_record(sample_index=2)])
        assert path.read_text(encoding="utf-8") == before
        assert [p.name for p in tmp_path.iterdir()] == ["census.jsonl"]


class TestCouplingClauseAccessors:
    """The stamp the rates carry so the control's 'invented coupling story' reading cannot land on a
    cell whose prompt stated the coupling."""

    def test_games_are_read_off_the_arm_table(self) -> None:
        for twin_framed in ("twin-pd", "public-goods", "stag-hunt", "fixed-pie-pd"):
            assert has_coupling_clause(twin_framed) is True, twin_framed
        for silent in ("pd-unstated", "pd-reskin"):
            assert has_coupling_clause(silent) is False, silent
        assert has_coupling_clause("pd-vs-frozen") is False, (
            "a recorded decision is a disclosure, not a coupling"
        )
        with pytest.raises(ValueError, match="not a 2x2 matrix game"):
            has_coupling_clause("dictator")

    def test_framings_are_read_off_the_clause_each_renders(self) -> None:
        for coupled in (
            FRAMING_TWIN,
            "different-ai-coupled",
            "stated-matcher",
            "stated-track-record",
            "stated-track-record-noisy",
            "stated-track-record-p60",
        ):
            assert framing_states_coupling(coupled) is True, coupled
        for decoupled in (
            "unstated",
            "human",
            "another-ai",
            "different-ai",
            "same-weights-uncorrelated",
            "stated-always-coop",
            "stated-always-defect",
        ):
            assert framing_states_coupling(decoupled) is False, decoupled
        with pytest.raises(ValueError, match="Unknown framing_id"):
            framing_states_coupling("FRAMING-SENTINEL")

    def test_every_registered_framing_classifies_and_the_twin_framing_agrees_with_the_twin_game(
        self,
    ) -> None:
        for framing_id in COUNTERPART_FRAMING_IDS:
            assert isinstance(framing_states_coupling(framing_id), bool)
        assert framing_states_coupling(FRAMING_TWIN) == has_coupling_clause("twin-pd")


class TestCLI:
    def test_judge_refuses_an_out_dir_git_would_track(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "arm-dir", "cell-y", 70, [battery_row()])
        tracked = REPO_ROOT / "games" / "zz-trace-judge-refused"
        with pytest.raises(ValueError, match="refusing to write a trace"):
            main(
                [
                    "judge",
                    "--tree",
                    str(tmp_path),
                    "--arm",
                    "arm-dir",
                    "--cell",
                    "cell-y",
                    "--steps",
                    "70",
                    "--out",
                    str(tracked),
                ]
            )
        assert not tracked.exists()

    def test_keyword_then_rates_end_to_end_offline(self, tmp_path: Path) -> None:
        rows = [battery_row(sample_index=i) for i in range(3)]
        write_cell(tmp_path, "arm-dir", "cell-y", 0, rows)
        write_cell(tmp_path, "arm-dir", "cell-y", 70, rows)
        out_dir = tmp_path / "out"
        main(
            [
                "keyword",
                "--tree",
                str(tmp_path),
                "--arm",
                "arm-dir",
                "--cell",
                "cell-y",
                "--steps",
                "0,70",
                "--out",
                str(out_dir),
            ]
        )
        keyword_rows = [
            json.loads(line) for line in (out_dir / "keyword.jsonl").read_text().splitlines()
        ]
        assert len(keyword_rows) == 6
        assert {row["keyword_basis"] for row in keyword_rows} == {"mirror"}
        census = load_census(out_dir / "census.jsonl")
        assert len(census) == 6
        # Rates over an empty judged file: every record is not_judged, and the file still renders.
        (out_dir / "judged.jsonl").write_text("")
        rates_path = out_dir / "rates.json"
        main(["rates", "--judged", str(out_dir / "judged.jsonl"), "--out", str(rates_path)])
        rates = json.loads(rates_path.read_text())
        label = f"arm-dir|cell-y|{SCOPE}@70"
        assert rates["cells"][label]["not_judged"] == 3
        assert rates["cells"][label]["judged"] == 0
