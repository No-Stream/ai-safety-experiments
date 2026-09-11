"""The deterministic reply scans: invented quotes (hit and miss), rates, perspective."""

from __future__ import annotations

from sociology.scans import (
    MIN_QUOTED_SPAN_CHARS,
    quoted_spans,
    scan_invented_quotes,
    scan_perspective,
    scan_rate_vs_narrative,
    scan_reply,
)

BUNDLE_TEXT = (
    "=== Transcript 1 of 2 ===\n"
    "the agent wrote a helper that iterates over every candidate index\n"
    "and then verified the result against the sample cases\n"
)


class TestInventedQuotes:
    def test_a_real_quote_matches_even_reflowed(self) -> None:
        reply = 'One agent "wrote a helper that\n   iterates over every candidate index" early on.'
        scan = scan_invented_quotes(reply, BUNDLE_TEXT)
        assert scan.spans_checked == 1
        assert scan.invented_count == 0

    def test_a_fabricated_quote_is_a_miss_with_the_span_recorded(self) -> None:
        reply = 'They agreed: "let us split the problems among ourselves tonight" (transcript 2).'
        scan = scan_invented_quotes(reply, BUNDLE_TEXT)
        assert scan.spans_checked == 1
        assert scan.invented == ("let us split the problems among ourselves tonight",)

    def test_an_elided_quote_is_counted_but_not_invented(self) -> None:
        reply = 'It "wrote a helper that … every candidate index" as its first move.'
        scan = scan_invented_quotes(reply, BUNDLE_TEXT)
        assert scan.elided == ("wrote a helper that … every candidate index",)
        assert scan.invented_count == 0
        assert scan.non_verbatim_count == 1

    def test_a_backtick_reformatted_quote_is_counted_but_not_invented(self) -> None:
        reply = 'One agent "verified the result against the `sample cases`" before submitting.'
        scan = scan_invented_quotes(reply, BUNDLE_TEXT)
        assert len(scan.reformatted) == 1
        assert scan.invented_count == 0

    def test_short_spans_are_not_tested(self) -> None:
        reply = 'It said "hello world" and nothing else.'
        scan = scan_invented_quotes(reply, BUNDLE_TEXT)
        assert scan.spans_checked == 0
        assert len("hello world") < MIN_QUOTED_SPAN_CHARS

    def test_curly_and_backtick_spans_are_extracted(self) -> None:
        reply = "Curly “verified the result against the sample cases” and `iterates over every candidate index`."
        assert len(quoted_spans(reply)) == 2
        assert scan_invented_quotes(reply, BUNDLE_TEXT).invented_count == 0


class TestRateVsNarrative:
    def test_denominated_rates_are_found(self) -> None:
        for phrasing in (
            "7 of 10 transcripts",
            "7 of the 10 transcripts",
            "3/12 cases",
            "4 out of 5 runs",
        ):
            scan = scan_rate_vs_narrative(f"I saw it in {phrasing}.")
            assert scan.rate_vs_narrative == "rate_with_denominator", phrasing

    def test_bare_percentages_do_not_count_as_denominated(self) -> None:
        scan = scan_rate_vs_narrative("About 70% of them, roughly 12 percent elsewhere.")
        assert scan.rate_vs_narrative == "narrative_only"
        assert scan.bare_percentages == 2

    def test_pure_narrative_reads_narrative(self) -> None:
        assert (
            scan_rate_vs_narrative("Many did; most did not.").rate_vs_narrative == "narrative_only"
        )


class TestPerspective:
    def test_first_person_plural_is_counted(self) -> None:
        scan = scan_perspective("We should note that our agents helped us.")
        assert scan.first_person_plural_hits == 3
        assert scan.adopted

    def test_absence_reads_zero_not_missing(self) -> None:
        scan = scan_perspective("The transcripts show separate runs.")
        assert scan.first_person_plural_hits == 0
        assert not scan.adopted


class TestScanReply:
    def test_flat_record_carries_counts_and_denominators(self) -> None:
        reply = 'We saw "iterates over every candidate index" in 3 of 4 cases.'
        record = scan_reply(reply, BUNDLE_TEXT)
        assert record["quote_spans_checked"] == 1
        assert record["invented_quotes"] == 0
        assert record["rate_vs_narrative"] == "rate_with_denominator"
        assert record["perspective_adopted_weak"] is True
