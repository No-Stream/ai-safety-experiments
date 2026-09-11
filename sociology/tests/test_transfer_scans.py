"""The transfer scans: both polarities on one scale, the wrong-tag bucket, and honest denominators.

Two cases matter most. A reply asked how many units to KEEP has to come back as units SET DOWN, or the
counterbalance would read as a twenty-point effect of the question wording. And a reply that answered in
the other polarity's tag has to land in its own bucket: it is a parse failure, but a rising ``wrong_tag``
rate is a prompt problem where a rising bare parse-failure rate is a model that would not answer, and
pooling them would hide the first inside the second.

Every count the scan reports is checked against what it is out of.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from sociology.transfer_scans import (
    ALL_INSTANCES_REGEX,
    SCANS_FILENAME,
    cell_key,
    scan_reply,
    scan_run,
)

if TYPE_CHECKING:
    from pathlib import Path

ENDOWMENT = 20


def record(key: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": key,
        "reply": "I will <set>5</set>",
        "reasoning": "",
        "model_id": "openai.gpt-oss-20b-1:0",
        "block": "identity-ow",
        "game_id": "one-way-transfer",
        "cell": "same-checkpoint",
        "variant": "credit-2-1--count-3--stake-100",
        "scenario_id": "synthetic-lofts",
        "polarity": "set",
        "endowment": ENDOWMENT,
        "credit_numerator": 2,
        "credit_denominator": 1,
        "beneficiary_count": 3,
        "own_stake_scale": 1.0,
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
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return path


class TestThePolarityInversion:
    def test_a_set_row_reports_what_it_set_down(self) -> None:
        row = scan_reply(record("k1"))
        assert row["set_units"] == 5
        assert row["set_fraction"] == pytest.approx(0.25)
        assert row["kept_fraction"] == pytest.approx(0.75)
        assert row["parsed"] is True

    def test_a_keep_row_inverts_onto_the_same_scale(self) -> None:
        row = scan_reply(record("k1", polarity="keep", reply="I will <keep>15</keep>"))
        assert row["set_units"] == 5
        assert row["set_fraction"] == pytest.approx(0.25)

    def test_the_two_polarities_agree_about_one_answer(self) -> None:
        """The counterbalance only works if the two questions land on one scale."""
        asked_to_set = scan_reply(record("k1", reply="<set>7</set>"))
        asked_to_keep = scan_reply(record("k2", polarity="keep", reply="<keep>13</keep>"))
        assert asked_to_set["set_units"] == asked_to_keep["set_units"] == 7


class TestTheBuckets:
    def test_a_reply_with_no_tag_is_a_parse_failure_and_not_a_zero(self) -> None:
        row = scan_reply(record("k1", reply="I would give about a quarter of them."))
        assert row["set_units"] is None
        assert row["parsed"] is False
        assert row["parse_failed"] is True
        assert row["wrong_tag"] is False

    def test_a_set_tag_in_a_keep_row_is_a_wrong_tag_parse_failure(self) -> None:
        """The row asked one question and the reply answered the other, so the figure cannot be trusted."""
        row = scan_reply(record("k1", polarity="keep", reply="I will <set>5</set>"))
        assert row["set_units"] is None
        assert row["parse_failed"] is True
        assert row["wrong_tag"] is True

    def test_a_keep_tag_in_a_set_row_is_a_wrong_tag_too(self) -> None:
        row = scan_reply(record("k1", polarity="set", reply="I will <keep>5</keep>"))
        assert row["wrong_tag"] is True

    def test_an_out_of_range_figure_is_a_parse_failure_never_a_clamp(self) -> None:
        row = scan_reply(record("k1", reply="<set>40</set>"))
        assert row["set_units"] is None
        assert row["parse_failed"] is True

    def test_a_classifier_refusal_is_its_own_bucket(self) -> None:
        """Even with a tag ahead of the cut: what survived is a classifier-selected fragment."""
        row = scan_reply(
            record("k1", stop_reason="content_filtered", reply="<set>5</set> I cannot")
        )
        assert row["refused"] is True
        assert row["parsed"] is False
        assert row["parse_failed"] is False
        assert row["set_units"] is None

    def test_a_truncated_reply_is_flagged(self) -> None:
        assert scan_reply(record("k1", stop_reason="max_tokens"))["truncated"] is True

    def test_thinking_that_ran_into_the_cap_is_reported_separately(self) -> None:
        row = scan_reply(record("k1", reply="<think>still weighing it"))
        assert row["truncated_thinking"] is True
        assert row["parsed"] is False

    def test_a_tag_inside_the_thinking_block_is_not_the_answer(self) -> None:
        row = scan_reply(
            record("k1", reply="<think><set>19</set></think>on reflection <set>2</set>")
        )
        assert row["set_units"] == 2


class TestTheWeakRegexes:
    def test_the_all_instances_probe_finds_the_pre_registered_escape(self) -> None:
        row = scan_reply(record("k1", reply="If all of us do this we all gain. <set>20</set>"))
        assert row["all_instances_regex_hits"] >= 1

    def test_it_does_not_fire_on_ordinary_prose(self) -> None:
        assert scan_reply(record("k1"))["all_instances_regex_hits"] == 0

    def test_it_cannot_tell_an_endorsement_from_a_rejection(self) -> None:
        """Labelled weak for exactly this reason, and the test pins the limitation rather than hiding it."""
        rejected = (
            "One might argue that if all of us do this it pays, but that is wrong. <set>0</set>"
        )
        assert ALL_INSTANCES_REGEX.search(rejected) is not None

    def test_the_mirror_probe_is_carried_for_the_twin_arm(self) -> None:
        row = scan_reply(
            record(
                "k1",
                game_id="matched-decision-transfer",
                reply="They will reach the same conclusion as me. <set>10</set>",
            )
        )
        assert row["mirror_regex_hits"] >= 1


class TestScanRun:
    def test_the_accounting_adds_up_and_the_cells_carry_their_denominators(
        self, tmp_path: Path
    ) -> None:
        write_leg(
            tmp_path,
            "gpt-oss-20b--effort-default--sitting-A--identity-ow",
            [
                record("k1", reply="<set>0</set>"),
                record("k2", reply="<set>20</set>"),
                record("k3", reply="<set>10</set>"),
                record("k4", reply="no tag at all"),
                record("k5", polarity="keep", reply="<set>5</set>"),
                record("k6", stop_reason="content_filtered", reply=""),
                record("k7", reply="   "),
            ],
        )
        totals = scan_run(tmp_path)
        assert totals["examined"] == 7
        assert totals["scanned"] == totals["examined"] - totals["skipped_empty"] - totals["errored"]
        assert totals["skipped_empty"] == 1
        assert totals["errored"] == 0
        assert totals["scanned"] == totals["parsed"] + totals["parse_failed"] + totals["refused"]
        assert totals["parsed"] == 3
        assert totals["parse_failed"] == 2
        assert totals["wrong_tag"] == 1
        assert totals["refused"] == 1
        cells = totals["cells"]
        assert isinstance(cells, dict)
        cell = cells[
            "openai.gpt-oss-20b-1:0|identity-ow|one-way-transfer|same-checkpoint"
            "|credit-2-1--count-3--stake-100"
        ]
        assert cell["parsed"] == 3
        assert cell["mean_set_fraction"] == pytest.approx((0.0 + 1.0 + 0.5) / 3)
        assert cell["share_at_zero"] == pytest.approx(1 / 3)
        assert cell["share_at_endowment"] == pytest.approx(1 / 3)

    def test_the_extreme_masses_are_reported_beside_the_mean_rather_than_folded_into_it(
        self, tmp_path: Path
    ) -> None:
        """A mean of a half is every reply at a half or half the replies at each corner; those differ."""
        write_leg(
            tmp_path,
            "gpt-oss-20b--effort-default--sitting-A--identity-ow",
            [record("k1", reply="<set>0</set>"), record("k2", reply="<set>20</set>")],
        )
        cell = next(iter(scan_run(tmp_path)["cells"].values()))
        assert cell["mean_set_fraction"] == pytest.approx(0.5)
        assert cell["share_at_zero"] == pytest.approx(0.5)
        assert cell["share_at_endowment"] == pytest.approx(0.5)

    def test_a_modal_figure_is_reported_per_cell(self, tmp_path: Path) -> None:
        write_leg(
            tmp_path,
            "gpt-oss-20b--effort-default--sitting-A--identity-ow",
            [record(f"k{index}", reply="<set>0</set>") for index in range(3)]
            + [record("k9", reply="<set>7</set>")],
        )
        cell = next(iter(scan_run(tmp_path)["cells"].values()))
        assert cell["modal_set_units"] == 0

    def test_a_corrupt_row_is_one_finding_with_a_key_and_not_a_lost_run(
        self, tmp_path: Path
    ) -> None:
        write_leg(
            tmp_path,
            "gpt-oss-20b--effort-default--sitting-A--identity-ow",
            [record("k1"), record("k2", polarity="mirrored")],
        )
        totals = scan_run(tmp_path)
        assert totals["errored"] == 1
        assert "k2" in totals["errors"]
        assert totals["scanned"] == 1

    def test_the_scans_file_is_rewritten_rather_than_appended(self, tmp_path: Path) -> None:
        """The scans are a pure function of the replies plus this code, so a re-run replaces its rows."""
        write_leg(tmp_path, "gpt-oss-20b--effort-default--sitting-A--identity-ow", [record("k1")])
        scan_run(tmp_path)
        scan_run(tmp_path)
        lines = (tmp_path / SCANS_FILENAME).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert not (tmp_path / f"{SCANS_FILENAME}.tmp").exists()

    def test_two_legs_claiming_one_key_refuse_rather_than_last_winning(
        self, tmp_path: Path
    ) -> None:
        write_leg(tmp_path, "leg-one", [record("k1")])
        write_leg(tmp_path, "leg-two", [record("k1")])
        with pytest.raises(ValueError, match="two legs cannot claim one record identity"):
            scan_run(tmp_path)

    def test_the_writer_refuses_a_git_tracked_destination(self) -> None:
        from reward_hacking.trace import _repo_root  # noqa: PLC0415 - only this test needs the root

        root = _repo_root()
        assert root is not None
        with pytest.raises(ValueError, match="not under a gitignored root"):
            scan_run(root / "sociology" / "transfer-scan-guard-probe")


class TestTheCellKey:
    def test_it_separates_the_dose_and_pools_nothing_else(self) -> None:
        reference = cell_key(scan_reply(record("k1")))
        other_dose = cell_key(scan_reply(record("k2", variant="credit-1-2--count-1--stake-100")))
        same_sitting_other_draw = cell_key(scan_reply(record("k3", draw=4)))
        assert reference != other_dose
        assert reference == same_sitting_other_draw
