"""The blind judge's offline machinery: prompt blindness, parsing, resume, retry, validation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from sociology import judge as judge_module
from sociology.judge import (
    SEVERITY_LEVELS,
    JudgeReplyError,
    JudgeVerdict,
    judge_prompt,
    judge_records,
    load_judged,
    parse_judge_reply,
    validate_judge,
)
from sociology.model_stub import ScriptedDetailedBackend
from sociology.tests.conftest import make_unit
from sociology.tests.streaming_stubs import (
    FrozenClock,
    LatencyScriptedBackend,
    LatencyStreamingBackend,
    ShortListBackend,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sociology.stimulus import Stimulus

VERDICT_PAYLOAD = {
    "severity": "role_differentiation",
    "uncertainty_flagged": False,
    "perspective_adoption": False,
    "rate_vs_narrative": "narrative_only",
    "evidence": "the agents divided the work",
}


def verdict_json(**overrides: object) -> str:
    return json.dumps({**VERDICT_PAYLOAD, **overrides})


def reply_record(key: str, reply: str = "an analysis reply") -> dict[str, object]:
    return {
        "key": key,
        "reply": reply,
        "cell": "center",
        "bundle_id": "abc123",
        "model_id": "scripted",
        "draw": 0,
        "reasoning_effort": None,
        "framing": "population",
        "cues": "kept",
        "family": "agentic-120b",
        "size": 10,
    }


class TestJudgePromptBlindness:
    def test_prompt_is_rubric_plus_reply_and_nothing_else(self, stimulus: Stimulus) -> None:
        prompt = judge_prompt("THE ANALYSIS REPLY", stimulus)
        assert prompt.startswith(stimulus.judge_instructions)
        assert "THE ANALYSIS REPLY" in prompt
        # Nothing that could tell the judge which cell produced the reply.
        for leak in ("population", "center", "bundle", "agentic-120b", "kept", "stripped"):
            assert leak not in prompt

    def test_bundle_text_never_reaches_the_judge(self, stimulus: Stimulus) -> None:
        unit = make_unit("ep-1", rendered="TRANSCRIPT BODY TEXT")
        prompt = judge_prompt("reply about patterns", stimulus)
        assert unit.rendered not in prompt


class TestParseJudgeReply:
    def test_valid_payload_parses(self) -> None:
        verdict = parse_judge_reply(verdict_json())
        assert verdict.severity == "role_differentiation"
        assert verdict.counts_toward_headline()

    def test_severity_order_is_the_ladder(self) -> None:
        ranks = [
            JudgeVerdict(
                severity=level,
                uncertainty_flagged=False,
                perspective_adoption=False,
                rate_vs_narrative="narrative_only",
                evidence="",
            ).severity_rank()
            for level in SEVERITY_LEVELS
        ]
        assert ranks == sorted(ranks)
        low = parse_judge_reply(verdict_json(severity="shared_idiom"))
        assert not low.counts_toward_headline()

    def test_fenced_json_is_found(self) -> None:
        assert (
            parse_judge_reply(f"```json\n{verdict_json()}\n```").severity == "role_differentiation"
        )

    @pytest.mark.parametrize(
        "broken",
        [
            "no json here at all",
            verdict_json(severity="catastrophic"),
            verdict_json(rate_vs_narrative="sometimes"),
            verdict_json(uncertainty_flagged="yes"),
            verdict_json(evidence=42),
        ],
    )
    def test_off_schema_replies_refuse(self, broken: str) -> None:
        with pytest.raises(JudgeReplyError):
            parse_judge_reply(broken)


class TestJudgeRecords:
    def test_rows_carry_provenance_and_verdicts(self, tmp_path: Path, stimulus: Stimulus) -> None:
        backend = ScriptedDetailedBackend([verdict_json()])
        out = tmp_path / "judged.jsonl"
        counts = judge_records(backend, [reply_record("k1")], out, stimulus)
        assert counts["judged"] == 1
        row = load_judged(out)["k1"]
        assert row["cell"] == "center"
        assert row["judge_prompt_version"] == "sociology-analysis-judge-v1"
        assert row["judge_prompt_digest"]
        assert row["verdict"]["severity"] == "role_differentiation"

    def test_resume_skips_already_judged_keys(self, tmp_path: Path, stimulus: Stimulus) -> None:
        out = tmp_path / "judged.jsonl"
        backend = ScriptedDetailedBackend([verdict_json()])
        judge_records(backend, [reply_record("k1")], out, stimulus)
        second = ScriptedDetailedBackend([verdict_json(severity="none")])
        counts = judge_records(second, [reply_record("k1"), reply_record("k2")], out, stimulus)
        assert counts["already_judged"] == 1
        assert counts["judged"] == 1
        rows = load_judged(out)
        assert rows["k1"]["verdict"]["severity"] == "role_differentiation", "k1 must not re-judge"
        assert rows["k2"]["verdict"]["severity"] == "none"

    def test_errored_rows_retry_once_and_last_wins(
        self, tmp_path: Path, stimulus: Stimulus
    ) -> None:
        backend = ScriptedDetailedBackend(["not json at all", verdict_json(severity="none")])
        out = tmp_path / "judged.jsonl"
        counts = judge_records(backend, [reply_record("k1")], out, stimulus)
        assert counts["errored_first_attempt"] == 1
        assert counts["errored_after_retry"] == 0
        assert load_judged(out)["k1"]["verdict"]["severity"] == "none"

    def test_truncated_judge_reply_is_an_error_never_a_verdict(
        self, tmp_path: Path, stimulus: Stimulus
    ) -> None:
        backend = ScriptedDetailedBackend([verdict_json()], stop_reason="max_tokens")
        out = tmp_path / "judged.jsonl"
        counts = judge_records(backend, [reply_record("k1")], out, stimulus, retry_errored=False)
        assert counts["errored_after_retry"] == 1
        assert "verdict" not in load_judged(out)["k1"]

    def test_empty_replies_are_skipped_and_counted(
        self, tmp_path: Path, stimulus: Stimulus
    ) -> None:
        backend = ScriptedDetailedBackend([verdict_json()])
        counts = judge_records(
            backend, [reply_record("k1", reply="  ")], tmp_path / "judged.jsonl", stimulus
        )
        assert counts["skipped_empty"] == 1
        assert counts["judged"] == 0


class TestValidateJudge:
    def test_misses_are_reported_by_name(self, tmp_path: Path, stimulus: Stimulus) -> None:
        # The synthetic stimulus registers v-none=none and v-coord=explicit_coordination; a judge
        # that answers "none" to both must produce exactly one named miss.
        backend = ScriptedDetailedBackend([verdict_json(severity="none")])
        report = validate_judge(backend, stimulus, tmp_path / "validation.jsonl")
        assert report["validated"] == 2
        assert report["agreed"] == 1
        assert report["misses"] == ["v-coord: expected explicit_coordination, got none"]


class TestTheJudgeKeepsTheQueueFullAndTheFileIdentical:
    """The same script through the streaming seam and the per-chunk path must write the same bytes.

    The scripted latencies make later prompts land FIRST on the streaming side, and every verdict
    quotes the key of the reply it answers, so a row paired by arrival order would carry another
    key's evidence and the file comparison would catch it on the first record. The clock is frozen
    so ``judged_at`` cannot be the one field that differs.
    """

    def records(self) -> list[dict[str, object]]:
        return [
            reply_record(f"k{index}", reply=f"analysis reply number {index}") for index in range(5)
        ]

    def scripts(self, stimulus: Stimulus) -> tuple[Callable[[str], str], Callable[[str], float]]:
        records = self.records()
        by_prompt = {judge_prompt(str(r["reply"]), stimulus): str(r["key"]) for r in records}
        latency = {prompt: float(len(by_prompt) - index) for index, prompt in enumerate(by_prompt)}
        return (lambda prompt: verdict_json(evidence=by_prompt[prompt])), latency.__getitem__

    def test_a_streaming_backend_writes_the_same_bytes_as_the_per_chunk_path(
        self, tmp_path: Path, stimulus: Stimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: rows paired by arrival order, or chunks released as they fill."""
        monkeypatch.setattr(judge_module, "datetime", FrozenClock)
        verdicts, latency = self.scripts(stimulus)
        per_chunk = tmp_path / "per-chunk.jsonl"
        streamed = tmp_path / "streamed.jsonl"
        reference = judge_records(
            LatencyScriptedBackend(verdicts, latency=latency),
            self.records(),
            per_chunk,
            stimulus,
            chunk_size=2,
        )
        counts = judge_records(
            LatencyStreamingBackend(verdicts, latency=latency),
            self.records(),
            streamed,
            stimulus,
            chunk_size=2,
        )
        assert counts == reference
        assert streamed.read_bytes() == per_chunk.read_bytes()
        rows = load_judged(streamed)
        assert list(rows) == [f"k{index}" for index in range(5)], "request order on disk"
        for key, row in rows.items():
            assert row["verdict"]["evidence"] == key, "each verdict answers its own reply"
            assert row["judge_elapsed_seconds"] == latency(
                judge_prompt(f"analysis reply number {key[1:]}", stimulus)
            )

    def test_the_finished_part_of_the_chunk_in_flight_lands_before_the_raise_and_resumes(
        self, tmp_path: Path, stimulus: Stimulus
    ) -> None:
        """SABOTAGE target: the finished siblings of a raising call discarded with the raise."""
        verdicts, latency = self.scripts(stimulus)
        records = self.records()
        bug = judge_prompt(str(records[2]["reply"]), stimulus)
        out = tmp_path / "judged.jsonl"
        with pytest.raises(RuntimeError, match="scripted request bug"):
            judge_records(
                LatencyStreamingBackend(verdicts, latency=latency, fail_on={bug}),
                records,
                out,
                stimulus,
                chunk_size=2,
            )
        assert sorted(load_judged(out)) == ["k0", "k1", "k3", "k4"]
        counts = judge_records(
            LatencyStreamingBackend(verdicts, latency=latency), records, out, stimulus, chunk_size=2
        )
        assert counts["already_judged"] == 4
        assert counts["judged"] == 1
        assert load_judged(out)["k2"]["verdict"]["evidence"] == "k2"

    def test_a_short_results_list_from_a_backend_without_a_stream_refuses_before_writing(
        self, tmp_path: Path, stimulus: Stimulus
    ) -> None:
        """SABOTAGE target: drop the length check and the rows land short while the pass reports whole."""
        verdicts, _ = self.scripts(stimulus)
        backend = ShortListBackend(verdicts)
        out = tmp_path / "judged.jsonl"
        with pytest.raises(RuntimeError, match="1 completions for the 2 prompts of chunk 1"):
            judge_records(backend, self.records(), out, stimulus, chunk_size=2)
        assert not out.exists(), "nothing of the short chunk was written"
        assert backend.calls == 1, "nothing past the short chunk was requested"
