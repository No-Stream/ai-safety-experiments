"""The runner's offline machinery: the leg plan, reply records, resume identity, size checks."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from sociology.bundles import build_manifest, write_manifest
from sociology.model_stub import ScriptedDetailedBackend
from sociology.runner import (
    ANALYSIS_MAX_TOKENS,
    HAIKU_MODEL_ID,
    LUNA_MODEL_ID,
    OPUS_MODEL_ID,
    SOL_MODEL_ID,
    PlannedCall,
    assert_context_fits,
    load_replies,
    planned_calls_for_model,
    record_key,
    run_live_calls,
)
from sociology.tests.test_bundles import synthetic_pools

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from sociology.corpus import BundleUnit
    from sociology.stimulus import Stimulus


def synthetic_units() -> dict[str, list[BundleUnit]]:
    return {family: list(pool.units.values()) for family, pool in synthetic_pools().items()}


@pytest.fixture
def manifest(tmp_path: Path, stimulus: Stimulus) -> dict[str, Any]:
    manifest = build_manifest(synthetic_pools(), stimulus)
    write_manifest(manifest, tmp_path)
    return manifest


class TestPlannedCalls:
    def test_call_counts_match_the_design_table(
        self, manifest: dict[str, Any], stimulus: Stimulus
    ) -> None:
        units = synthetic_units()
        expected = {OPUS_MODEL_ID: 396, HAIKU_MODEL_ID: 180, SOL_MODEL_ID: 40, LUNA_MODEL_ID: 40}
        for model_id, count in expected.items():
            calls = planned_calls_for_model(model_id, manifest, units, stimulus)
            assert len(calls) == count, model_id
            assert len({call.key for call in calls}) == count, "keys must be unique"

    def test_luna_efforts_are_distinct_keys_on_the_same_bundles(
        self, manifest: dict[str, Any], stimulus: Stimulus
    ) -> None:
        calls = planned_calls_for_model(LUNA_MODEL_ID, manifest, synthetic_units(), stimulus)
        by_effort: dict[str | None, set[str]] = {}
        for call in calls:
            by_effort.setdefault(call.reasoning_effort, set()).add(call.bundle_id)
        assert set(by_effort) == {"medium", "high"}
        assert by_effort["medium"] == by_effort["high"], (
            "the effort rungs must read matched bundles"
        )

    def test_framing_and_cues_ride_the_cell_not_the_bundle(
        self, manifest: dict[str, Any], stimulus: Stimulus
    ) -> None:
        calls = planned_calls_for_model(OPUS_MODEL_ID, manifest, synthetic_units(), stimulus)
        by_cell: dict[str, PlannedCall] = {}
        for call in calls:
            by_cell.setdefault(call.cell, call)
        assert by_cell["framing-independent"].framing == "independent"
        assert by_cell["cues-stripped"].cues == "stripped"
        # Matched design: the same bundle id appears under both framings with different prompts.
        center = next(c for c in calls if c.cell == "center")
        twin = next(
            c for c in calls if c.cell == "framing-independent" and c.bundle_id == center.bundle_id
        )
        assert twin.prompt != center.prompt

    def test_unknown_model_refuses(self, manifest: dict[str, Any], stimulus: Stimulus) -> None:
        with pytest.raises(ValueError, match="no production legs"):
            planned_calls_for_model("nonexistent-model", manifest, synthetic_units(), stimulus)


def planned_call(key_suffix: str, prompt: str = "PROMPT") -> PlannedCall:
    return PlannedCall(
        key=record_key("center", f"bundle-{key_suffix}", "scripted", None, 0),
        cell="center",
        bundle_id=f"bundle-{key_suffix}",
        family="agentic-120b",
        framing="population",
        cues="kept",
        size=10,
        model_id="scripted",
        transport="live",
        draw=0,
        reasoning_effort=None,
        prompt=prompt,
    )


def scripted_factory(
    *replies: str, stop_reason: str = "end_turn"
) -> Callable[[str, str | None, int], ScriptedDetailedBackend]:
    """A backend factory serving the given script, so no test builds a real Bedrock client."""

    def factory(model_id: str, effort: str | None, concurrency: int) -> ScriptedDetailedBackend:
        return ScriptedDetailedBackend(list(replies), stop_reason=stop_reason)

    return factory


class TestRunLiveResume:
    def test_resume_skips_completed_keys_and_counts_them_separately(
        self, tmp_path: Path, stimulus: Stimulus
    ) -> None:
        out = tmp_path / "replies.jsonl"
        calls = [planned_call("a"), planned_call("b")]
        factory = scripted_factory("an analysis reply")
        first = run_live_calls(
            calls, out, stimulus, concurrency=1, chunk_size=1, backend_factory=factory
        )
        assert first == {"planned": 2, "resumed": 0, "ran": 2, "incomplete": 0}
        # A second invocation with one extra call must skip the two on disk, not re-run them.
        second = run_live_calls(
            [*calls, planned_call("c")],
            out,
            stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=factory,
        )
        assert second == {"planned": 3, "resumed": 2, "ran": 1, "incomplete": 0}
        rows = load_replies(out)
        assert sorted(rows) == sorted(planned_call(suffix).key for suffix in ("a", "b", "c"))

    def test_incomplete_records_are_flagged_kept_and_counted(
        self, tmp_path: Path, stimulus: Stimulus
    ) -> None:
        out = tmp_path / "replies.jsonl"
        backend_calls = [planned_call("a")]
        counts = run_live_calls(
            backend_calls,
            out,
            stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=lambda model_id, effort, concurrency: ScriptedDetailedBackend(
                ["partial text"], stop_reason="call_failed:ReadTimeoutError"
            ),
        )
        assert counts["incomplete"] == 1
        row = load_replies(out)[backend_calls[0].key]
        assert row["incomplete"] is True
        assert row["reply"] == "partial text", "a partial record keeps what arrived"

    def test_duplicate_reply_keys_refuse_at_load(self, tmp_path: Path, stimulus: Stimulus) -> None:
        out = tmp_path / "replies.jsonl"
        run_live_calls(
            [planned_call("a")],
            out,
            stimulus,
            concurrency=1,
            chunk_size=1,
            backend_factory=scripted_factory("a reply"),
        )
        line = out.read_text(encoding="utf-8")
        out.write_text(line + line, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate reply key"):
            load_replies(out)


class TestContextCheck:
    def test_oversized_prompt_refuses_before_any_spend(self) -> None:
        oversized = replace(planned_call("big", prompt="x" * 3_000_000), model_id=OPUS_MODEL_ID)
        with pytest.raises(RuntimeError, match="200000-token window"):
            assert_context_fits(OPUS_MODEL_ID, [oversized])

    def test_fitting_prompt_records_the_check(self) -> None:
        call = replace(planned_call("small", prompt="x" * 1000), model_id=OPUS_MODEL_ID)
        check = assert_context_fits(OPUS_MODEL_ID, [call])
        assert check["fits"] is True
        assert check["estimated_tokens_with_reply"] >= ANALYSIS_MAX_TOKENS
