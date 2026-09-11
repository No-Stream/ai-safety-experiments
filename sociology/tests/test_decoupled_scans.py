"""The ladder's deterministic scans: parse outcomes, truncation flags, and honest denominators.

The parse-failure case is the one that matters most. A reply that argues its way to a label without
emitting the tag must come back as ``action=None`` with ``cooperated=None``, never clamped into
either bucket -- the identity effect this pass is measuring is a few points wide, so a clamp would
manufacture one. Every count the scan reports is checked against what it is out of.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from sociology.decoupled_scans import (
    MIRROR_REGEX,
    SCANS_FILENAME,
    STOP_REASON_CONTENT_FILTERED,
    is_refused,
    load_run_replies,
    refuse_pooled_stimulus_mixture,
    scan_reply,
    scan_run,
)

if TYPE_CHECKING:
    from pathlib import Path

LABEL_A = "SHORT"
LABEL_B = "LONG"


def record(key: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": key,
        "reply": f"I will go with <action>{LABEL_A}</action>",
        "reasoning": "",
        "label_a": LABEL_A,
        "label_b": LABEL_B,
        "coop_label": LABEL_A,
        "model_id": "openai.gpt-oss-20b-1:0",
        "cell": "twin",
        "block": "anchor",
        "game_id": "twin-pd",
        "prompt_id": f"twin-pd--x--temptation-2--framing-twin--coop0--{key}",
        "label_print_order": "canonical",
        "reasoning_effort": None,
        "sitting": "A",
        "draw": 0,
        "stop_reason": "end_turn",
        "incomplete": False,
        "output_tokens": 120,
        "stimulus_digest": "0123456789abcdef",
    }
    row.update(overrides)
    return row


def write_leg(run_dir: Path, stem: str, records: list[dict[str, Any]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"replies--{stem}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return path


class TestScanReply:
    def test_a_tagged_cooperative_reply_scans_as_cooperation(self) -> None:
        row = scan_reply(record("k1"))
        assert row["parsed"] is True
        assert row["action"] == "cooperate"
        assert row["cooperated"] is True

    def test_the_other_label_scans_as_the_other_action(self) -> None:
        row = scan_reply(record("k1", reply=f"<action>{LABEL_B}</action>"))
        assert row["action"] == "defect"
        assert row["cooperated"] is False

    def test_a_reply_without_a_tag_is_unparsed_and_never_clamped(self) -> None:
        row = scan_reply(record("k1", reply="I would rather keep my options open."))
        assert row["parsed"] is False
        assert row["action"] is None
        assert row["cooperated"] is None

    def test_unclosed_thinking_is_reported_separately_from_a_plain_non_answer(self) -> None:
        row = scan_reply(record("k1", reply="<think>still weighing the table"))
        assert row["truncated_thinking"] is True
        assert row["parsed"] is False

    def test_a_capped_reply_is_flagged_as_truncated(self) -> None:
        row = scan_reply(record("k1", stop_reason="max_tokens"))
        assert row["truncated"] is True

    def test_thinking_is_stripped_before_the_answer_is_read(self) -> None:
        reply = f"<think>maybe {LABEL_B}</think>final: <action>{LABEL_A}</action>"
        row = scan_reply(record("k1", reply=reply))
        assert row["action"] == "cooperate"
        assert row["truncated_thinking"] is False

    def test_the_mirror_regex_counts_hits_and_is_only_a_lead(self) -> None:
        row = scan_reply(
            record(
                "k1",
                reply=f"it will do the same as me by the same reasoning <action>{LABEL_A}</action>",
            )
        )
        assert row["mirror_regex_hits"] >= 1
        assert scan_reply(record("k2"))["mirror_regex_hits"] == 0

    def test_the_mirror_regex_cannot_tell_endorsement_from_rejection(self) -> None:
        """Documented weakness, pinned so nobody reads its count as a mirror rate."""
        rejected = "one might think it will do the same as me, but that is wrong"
        assert len(MIRROR_REGEX.findall(rejected)) == 1

    def test_a_corrupt_label_triple_raises_rather_than_scoring(self) -> None:
        with pytest.raises(ValueError, match="coop_label"):
            scan_reply(record("k1", coop_label="EARTH"))

    def test_a_content_filtered_reply_is_refused_and_neither_parsed_nor_a_failure(self) -> None:
        """The classifier's refusal is a bucket of its own: the model never answered."""
        row = scan_reply(record("k1", reply="", stop_reason=STOP_REASON_CONTENT_FILTERED))
        assert is_refused(record("k1", stop_reason=STOP_REASON_CONTENT_FILTERED))
        assert row["refused"] is True
        assert row["parsed"] is False
        assert row["action"] is None
        assert row["cooperated"] is None
        assert row["truncated"] is False

    def test_a_tag_that_survived_the_classifiers_cut_is_still_not_an_answer(self) -> None:
        """A mid-reply refusal leaves a fragment the classifier selected, not the model's decision."""
        row = scan_reply(
            record(
                "k1",
                reply=f"<action>{LABEL_A}</action> and here is why",
                stop_reason=STOP_REASON_CONTENT_FILTERED,
            )
        )
        assert row["refused"] is True
        assert row["parsed"] is False
        assert row["action"] is None

    def test_a_refused_row_with_corrupt_labels_still_errors(self) -> None:
        """`errored` means one thing in every bucket, so the label triple is checked before the bucket."""
        with pytest.raises(ValueError, match="coop_label"):
            scan_reply(record("k1", coop_label="EARTH", stop_reason=STOP_REASON_CONTENT_FILTERED))

    def test_an_ordinary_reply_is_not_refused(self) -> None:
        assert scan_reply(record("k1"))["refused"] is False
        assert scan_reply(record("k1", stop_reason="max_tokens"))["refused"] is False


class TestLoadRunReplies:
    def test_every_legs_file_is_merged(self, tmp_path: Path) -> None:
        write_leg(tmp_path, "model-a--effort-default--sitting-A--anchor", [record("k1")])
        write_leg(tmp_path, "model-a--effort-default--sitting-B--floor", [record("k2")])
        assert sorted(load_run_replies(tmp_path)) == ["k1", "k2"]

    def test_a_key_claimed_by_two_legs_refuses(self, tmp_path: Path) -> None:
        write_leg(tmp_path, "model-a--effort-default--sitting-A--anchor", [record("k1")])
        write_leg(tmp_path, "model-b--effort-default--sitting-A--anchor", [record("k1")])
        with pytest.raises(ValueError, match="two legs cannot claim one record identity"):
            load_run_replies(tmp_path)

    def test_a_duplicate_key_inside_one_leg_refuses(self, tmp_path: Path) -> None:
        write_leg(
            tmp_path, "model-a--effort-default--sitting-A--anchor", [record("k1"), record("k1")]
        )
        with pytest.raises(ValueError, match="duplicate reply key"):
            load_run_replies(tmp_path)


class TestThePooledStimulusRefusal:
    def test_one_stimulus_passes_and_returns_its_digest(self, tmp_path: Path) -> None:
        write_leg(tmp_path, "model-a--effort-default--sitting-A--anchor", [record("k1")])
        replies = load_run_replies(tmp_path)
        assert refuse_pooled_stimulus_mixture(replies, run_dir=tmp_path, what="the judge") == (
            "0123456789abcdef"
        )

    def test_two_stimuli_in_one_run_directory_refuse_with_both_counts(self, tmp_path: Path) -> None:
        """Reachable by editing a frame between two legs: the live resume guard sees no shared key."""
        write_leg(tmp_path, "model-a--effort-default--sitting-A--anchor", [record("k1")])
        write_leg(
            tmp_path,
            "model-b--effort-default--sitting-A--anchor",
            [record("k2", stimulus_digest="ffffffffffffffff")],
        )
        replies = load_run_replies(tmp_path)
        with pytest.raises(ValueError, match="different stimulus prompt digests") as raised:
            refuse_pooled_stimulus_mixture(replies, run_dir=tmp_path, what="the judge")
        assert "0123456789abcdef" in str(raised.value)
        assert "ffffffffffffffff" in str(raised.value)

    def test_the_scan_is_not_gated_so_an_operator_can_see_which_leg_is_which(
        self, tmp_path: Path
    ) -> None:
        """Its rows carry the digest one per row, which is how the mixture is diagnosed at all."""
        write_leg(tmp_path, "model-a--effort-default--sitting-A--anchor", [record("k1")])
        write_leg(
            tmp_path,
            "model-b--effort-default--sitting-A--anchor",
            [record("k2", stimulus_digest="ffffffffffffffff")],
        )
        totals = scan_run(tmp_path)
        assert totals["scanned"] == 2
        digests = {
            json.loads(line)["stimulus_digest"]
            for line in (tmp_path / SCANS_FILENAME).read_text(encoding="utf-8").splitlines()
        }
        assert digests == {"0123456789abcdef", "ffffffffffffffff"}


class TestScanRun:
    def test_the_denominators_account_for_every_record_on_disk(self, tmp_path: Path) -> None:
        """One record of every class: parsed twice over, a parse failure, an empty, a corrupt row, a
        capped one, and two classifier refusals (no text, and a fragment)."""
        run_dir = tmp_path / "run"
        write_leg(
            run_dir,
            "model-a--effort-default--sitting-A--anchor",
            [
                record("k1"),
                record("k2", reply=f"<action>{LABEL_B}</action>"),
                record("k3", reply="no tag here"),
                record("k4", reply="   "),
                record("k5", coop_label="EARTH"),
                record("k6", stop_reason="max_tokens"),
                record("k7", reply="", stop_reason=STOP_REASON_CONTENT_FILTERED),
                record("k8", reply="I would rather", stop_reason=STOP_REASON_CONTENT_FILTERED),
            ],
        )
        totals = scan_run(run_dir)
        assert totals["examined"] == 8
        assert totals["skipped_empty"] == 1
        assert totals["errored"] == 1
        assert totals["scanned"] == 6
        assert totals["scanned"] + totals["skipped_empty"] + totals["errored"] == totals["examined"]
        assert totals["refused"] == 2
        assert totals["parsed"] == 3
        assert totals["parse_failed"] == 1
        assert totals["parsed"] + totals["parse_failed"] + totals["refused"] == totals["scanned"]
        assert totals["cooperated"] == 2
        assert totals["truncated"] == 1
        errors = totals["errors"]
        assert isinstance(errors, dict)
        assert "k5" in errors
        rows = {
            json.loads(line)["key"]: json.loads(line)
            for line in (run_dir / "scans.jsonl").read_text(encoding="utf-8").splitlines()
        }
        assert "k4" not in rows, (
            "an empty reply the classifier did not refuse is skipped, not written"
        )
        assert rows["k7"]["refused"] is True
        assert rows["k8"]["refused"] is True
        assert rows["k3"]["refused"] is False
        assert rows["k3"]["parsed"] is False

    def test_the_scans_file_is_rewritten_rather_than_appended(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        write_leg(run_dir, "model-a--effort-default--sitting-A--anchor", [record("k1")])
        scan_run(run_dir)
        scan_run(run_dir)
        lines = (run_dir / "scans.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["key"] == "k1"
        assert not list(run_dir.glob("scans.jsonl.tmp")), "the staging file is renamed into place"

    def test_an_empty_run_dir_scans_to_zero_with_its_denominators(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        totals = scan_run(run_dir)
        assert totals["examined"] == 0
        assert totals["scanned"] == 0
