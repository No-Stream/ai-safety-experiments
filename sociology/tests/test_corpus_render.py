"""Rendering: reasoning interleave anchoring, the cue scrub and its gate, and metadata blindness.

All transcripts here are synthetic, built the way the harness builds real ones (each turn's reply
entering as ``"\\n" + completion + "\\n"``), so the anchoring rules are exercised without a byte of
real transcript text in a tracked file.
"""

from __future__ import annotations

import pytest

from sociology.bundles import CUES_KEPT, CUES_STRIPPED, render_bundle_text
from sociology.corpus import (
    PRIVATE_REASONING_CLOSE,
    PRIVATE_REASONING_EMPTY_OPEN,
    PRIVATE_REASONING_OPEN,
    EpisodeTurn,
    assert_cue_free,
    contains_colocation_cue,
    render_episode,
    render_single_turn,
    scrub_colocation_cues,
    twin_key,
)
from sociology.tests.conftest import make_unit

LIMITER_LINE = "resource-limits: unit=reslimit-123456-episode_jail.sh cpus=4/64 mem_max=8G\n"


def harness_transcript(prompt: str, *turns: tuple[str, str]) -> tuple[str, tuple[EpisodeTurn, ...]]:
    """Build (transcript, turns) the way the harness does: reply as newline-wrapped, then tail."""
    transcript = prompt
    episode_turns: list[EpisodeTurn] = []
    for index, (completion, tail) in enumerate(turns):
        transcript += "\n" + completion + "\n"
        transcript += tail
        episode_turns.append(
            EpisodeTurn(index=index, completion=completion, reasoning=f"thought-{index}")
        )
    return transcript, tuple(episode_turns)


class TestRenderEpisode:
    def test_reasoning_lands_immediately_before_each_reply(self) -> None:
        transcript, turns = harness_transcript(
            "TASK PROMPT",
            ("reply-zero runs a command", "<result>ok</result>\n"),
            ("reply-one submits", ""),
        )
        rendered = render_episode(transcript, turns, unit_id="ep-0")
        for index, marker in ((0, "reply-zero"), (1, "reply-one")):
            block = f"{PRIVATE_REASONING_OPEN}\nthought-{index}\n{PRIVATE_REASONING_CLOSE}"
            assert block in rendered
            assert rendered.index(block) < rendered.index(marker)
        # The observation tail stays between turn 0's reply and turn 1's reasoning.
        assert rendered.index("<result>ok</result>") < rendered.index("thought-1")

    def test_empty_reasoning_inserts_no_block(self) -> None:
        transcript = "PROMPT\nreply\n"
        turns = (EpisodeTurn(index=0, completion="reply", reasoning=""),)
        rendered = render_episode(transcript, turns, unit_id="ep-0")
        assert PRIVATE_REASONING_OPEN not in rendered

    def test_trailing_empty_reply_appends_its_labelled_reasoning_at_the_end(self) -> None:
        transcript, turns = harness_transcript("PROMPT", ("real reply", "<result>x</result>\n"))
        transcript += "\n\n"  # the final empty reply, as the harness appends it
        all_turns = (*turns, EpisodeTurn(index=1, completion="", reasoning="final-thought"))
        rendered = render_episode(transcript, all_turns, unit_id="ep-0")
        assert rendered.rstrip().endswith(PRIVATE_REASONING_CLOSE)
        assert "final-thought" in rendered
        assert PRIVATE_REASONING_EMPTY_OPEN in rendered
        assert rendered.index("final-thought") > rendered.index("real reply")

    def test_mid_episode_empty_reply_reasoning_is_kept_labelled_and_ordered(self) -> None:
        # The nudged-empty-start shape: an empty turn 0, the nudge, then a real turn 1. The empty
        # turn's reasoning is carried to the next anchor under the empty-reply label, still ahead
        # of turn 1's own reasoning and reply.
        transcript = "PROMPT" + "\n\n" + "\nNUDGE TEXT\n" + "\nreal reply\n"
        turns = (
            EpisodeTurn(index=0, completion="", reasoning="empty-start-thought"),
            EpisodeTurn(index=1, completion="real reply", reasoning="second-thought"),
        )
        rendered = render_episode(transcript, turns, unit_id="ep-0")
        assert PRIVATE_REASONING_EMPTY_OPEN in rendered
        assert rendered.index("empty-start-thought") < rendered.index("second-thought")
        assert rendered.index("second-thought") < rendered.index("real reply")

    def test_unanchorable_completion_raises_with_the_unit_id(self) -> None:
        turns = (EpisodeTurn(index=0, completion="never appears", reasoning="x"),)
        with pytest.raises(ValueError, match="ep-broken"):
            render_episode("PROMPT with no reply", turns, unit_id="ep-broken")


class TestSingleTurnRender:
    def test_reasoning_then_completion_labelled(self) -> None:
        rendered = render_single_turn("the visible answer", "the private thought")
        assert rendered.index(PRIVATE_REASONING_OPEN) < rendered.index("the private thought")
        assert rendered.index("the private thought") < rendered.index("the visible answer")

    def test_no_reasoning_renders_completion_alone(self) -> None:
        rendered = render_single_turn("just the answer", "")
        assert PRIVATE_REASONING_OPEN not in rendered
        assert "just the answer" in rendered


class TestTwinKey:
    def test_conflicting_and_original_share_a_key(self) -> None:
        assert twin_key("ilcb-conflicting-lcbhard_16") == twin_key("ilcb-original-lcbhard_16")

    def test_different_indices_differ(self) -> None:
        assert twin_key("ilcb-original-lcbhard_16") != twin_key("ilcb-original-lcbhard_17")

    def test_out_of_scheme_id_is_its_own_key(self) -> None:
        assert twin_key("some-other-task") == "some-other-task"


class TestCueScrubAndGate:
    def test_scrub_removes_the_limiter_line_and_gate_passes(self) -> None:
        text = f"<stderr>{LIMITER_LINE}</stderr>"
        assert contains_colocation_cue(text)
        scrubbed = scrub_colocation_cues(text)
        assert scrubbed == "<stderr></stderr>"
        assert_cue_free(scrubbed, context="test")

    def test_gate_goes_red_on_a_planted_limiter_line(self) -> None:
        # The sabotage the gate exists for: a stripped rendering still carrying the cue.
        with pytest.raises(ValueError, match="stripped rendering still contains"):
            assert_cue_free(f"clean text\n{LIMITER_LINE}more text", context="test")

    def test_gate_goes_red_on_a_host_path_marker(self) -> None:
        with pytest.raises(ValueError, match="stripped rendering still contains"):
            assert_cue_free("output mentioning /tmp/agent-harness-abc/work", context="test")

    def test_stripped_bundle_rendering_scrubs_and_kept_keeps(self) -> None:
        unit = make_unit("ep-cue", rendered=f"body with cue\n{LIMITER_LINE}tail")
        kept = render_bundle_text([unit], cues=CUES_KEPT, context="kept")
        stripped = render_bundle_text([unit], cues=CUES_STRIPPED, context="stripped")
        assert contains_colocation_cue(kept), (
            "the kept rung must still carry the cue: it is the factor"
        )
        assert not contains_colocation_cue(stripped)


class TestBundleRenderBlindness:
    def test_rendered_bundle_carries_no_metadata_identifiers(self) -> None:
        units = [
            make_unit(
                "openai.gpt-oss-120b-1:0:ilcb-original-lcbhard_1:check+hidden:abc123.000",
                rendered="synthetic body text one",
            ),
            make_unit(
                "openai.gpt-oss-120b-1:0:ilcb-original-lcbhard_2:check+hidden:def456.000",
                rendered="synthetic body text two",
            ),
        ]
        text = render_bundle_text(units, cues=CUES_KEPT, context="blindness")
        for marker in (
            units[0].unit_id,
            "openai.gpt-oss-120b-1:0",
            "check+hidden",
            "episode_dir",
            "/tmp/agent-harness",  # noqa: S108 - a leak marker under test, not a path in use
        ):
            assert marker not in text
        assert "=== Transcript 1 of 2 ===" in text
        assert "=== Transcript 2 of 2 ===" in text
