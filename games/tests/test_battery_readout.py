"""Tests for the self-updating battery readout.

Every fixture here is synthetic and built in the test. Real trace data is never committed: an eval
cell carries the model's own completions, and the corpus items inside them are benchmark material.

The fixtures are deliberately tiny -- four game prompts rather than a hundred and seventy -- because
what is under test is the plumbing: which denominator lands in a cell, which banner fires, which
cell gets excluded. The numbers are chosen to make each of those visible, not to resemble a run.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import pytest

from games import battery_cells, battery_readout, battery_tables
from games.evals import (
    RECORD_META,
    SECTION_CAPABILITIES,
    SECTION_DT_PROBES,
    SECTION_GAME_BEHAVIOR,
    SECTION_SELF_REPORT,
)
from games.prompts import EVAL_ONLY_GAME_IDS
from games.survey import (
    COUNTERPART_AI,
    COUNTERPART_ARM_MISSING,
    COUNTERPART_HUMAN,
    COUNTERPART_NOMINAL_ANSWER,
    COUNTERPART_NOTHING_PARSED,
    COUNTERPART_UNSPECIFIED,
    FAMILY_GRADED_DIMENSION_AWARENESS,
    FAMILY_NEGATIVE_CONTROL,
    FAMILY_VALUES_FORCED_CHOICE,
    NON_SOCIAL_LABEL_PREFIX,
    SURVEY_CHEAP_TALK,
    SURVEY_CHOICE,
    SURVEY_ORDERED_CHOICE,
    SURVEY_TAGGED,
    TAGGED_ITEM_MISSING,
    TAGGED_NOTHING_PARSED,
    TAGGED_VOCABULARY_WIDTH_DIFFERS,
    WORDING_AS_PUBLISHED,
)

# Real registry entries: the readout looks each arm's trained game up in games.arms.ARMS.
GROUP_ARM = "twin-pd-group"
SELF_ARM = "twin-pd-self"
RUNG_ARM = "stag-hunt-safe-rung"
DICTATOR_ARM = "dictator"
UNREGISTERED_ARM = "arm-that-was-never-registered"

TWIN_GAME = "twin-pd"
STAG_GAME = "stag-hunt"
DICTATOR_GAME = "dictator"
NEGATIVE_CONTROL = battery_tables.NEGATIVE_CONTROL_GAME_ID

PINNED_RUNG = "safe-hunt"
OTHER_RUNG = "risky-hunt"

# Unmistakable in a document, so the no-generated-text assertion cannot pass by accident.
SENTINEL_COMPLETION = "SENTINEL-THINKING-TEXT-THAT-MUST-NEVER-REACH-A-TABLE"

SAMPLING: dict[str, Any] = {"temperature": 1.0, "presence_penalty": 1.5}


def meta_record(arm: str, step: int, **overrides: Any) -> dict[str, Any]:
    """One trace's meta record: what says which arm and checkpoint produced the rows."""
    record: dict[str, Any] = {
        "record": RECORD_META,
        "arm": arm,
        "step": step,
        "written_at": f"2026-08-20T0{step // 10}:00:00+00:00",
        "git_sha": "abc1234",
        "backend_kind": "vllm",
        "thinking": True,
        "sampling": dict(SAMPLING),
        "grading": "group-mix",
    }
    return record | overrides


def game_row(  # noqa: PLR0913 - one record's worth of labels, not a bundle worth naming
    game_id: str,
    prompt_id: str,
    *,
    reskin_id: str = "frame-one",
    payoff_variant: str = "temptation-2",
    coop: float | None = 1.0,
    truncated: bool = False,
    coop_label: str = "SHORT",
    label_print_order: str = "canonical",
) -> dict[str, Any]:
    """One matrix-game behaviour record; `coop=None` means it did not parse.

    `coop_label` is one of the two authored labels (`SHORT` first, `LONG` second) and
    `label_print_order` is the order the prompt printed them in, both as the battery stamps them.
    """
    return {
        "record": SECTION_GAME_BEHAVIOR,
        "game_id": game_id,
        "prompt_id": prompt_id,
        "reskin_id": reskin_id,
        "payoff_variant": payoff_variant,
        "render_grading": "group-mix",
        "label_a": "SHORT",
        "label_b": "LONG",
        "coop_label": coop_label,
        "label_print_order": label_print_order,
        "trained_game": False,
        "eval_only_game": game_id in EVAL_ONLY_GAME_IDS,
        "truncated_thinking": truncated,
        "completion": SENTINEL_COMPLETION,
        "visible_text": SENTINEL_COMPLETION,
        "action": None if coop is None else ("C" if coop else "D"),
        "coop_fraction": coop,
        "parsed": coop is not None,
    }


def dictator_row(
    prompt_id: str, *, keep: float | None = 0.5, endowment: int = 100
) -> dict[str, Any]:
    """One unilateral-split record, whose behaviour field is the fraction kept."""
    return {
        "record": SECTION_GAME_BEHAVIOR,
        "game_id": DICTATOR_GAME,
        "prompt_id": prompt_id,
        "reskin_id": "frame-one",
        "payoff_variant": f"endowment-{endowment}",
        "render_grading": "keep-fraction",
        "label_a": "KEEP",
        "label_b": "GIVE",
        "coop_label": "GIVE",
        "trained_game": False,
        "eval_only_game": False,
        "truncated_thinking": False,
        "completion": SENTINEL_COMPLETION,
        "visible_text": SENTINEL_COMPLETION,
        "kept": None if keep is None else int(keep * endowment),
        "keep_fraction": keep,
        "parsed": keep is not None,
    }


def iterated_row(prompt_id: str, moves: list[str]) -> dict[str, Any]:
    """One iterated-game record, whose per-round moves drive the round profile."""
    row = game_row("iterated-pd-tft", prompt_id)
    row["moves"] = moves
    row["coop_fraction"] = moves.count("C") / len(moves)
    row["action"] = None
    return row


def open_ended_probe(probe_id: str, theory: str, *, sample_index: int = 0) -> dict[str, Any]:
    """One free-response decision-theory record, scored by which theory its tag named."""
    return {
        "record": SECTION_DT_PROBES,
        "probe_id": probe_id,
        "source": "ours",
        "family": "open-ended",
        "kind": "open-ended",
        "sample_index": sample_index,
        "option_order_name": "no-options",
        "option_order": [],
        "truncated_thinking": False,
        "completion": SENTINEL_COMPLETION,
        "visible_text": SENTINEL_COMPLETION,
        "theory": theory,
        "parsed": True,
    }


def choice_probe(  # noqa: PLR0913 - one record's worth of labels, not a bundle worth naming
    probe_id: str,
    *,
    answer_index: int | None,
    order_name: str,
    edt_leaning: float | None,
    prosocial: bool | None = True,
    compatible: list[str] | None = None,
) -> dict[str, Any]:
    """One multiple-choice decision-theory record, asked under one of its two option orders.

    `answer_index=None` means the render did not parse, which on a real trace nulls every scored
    field (the item's valence included). `compatible` overrides the theories the chosen option is
    compatible with; the default derives one from the leaning.
    """
    unparsed = answer_index is None
    if not unparsed and compatible is None:
        compatible = ["EDT"] if edt_leaning else ["CDT"]
    return {
        "record": SECTION_DT_PROBES,
        "probe_id": probe_id,
        "source": "ours",
        "family": "twin",
        "kind": "multiple-choice",
        "sample_index": 0,
        "option_order_name": order_name,
        "option_order": [0, 1],
        "answer_index": answer_index,
        "presented_answer_index": answer_index,
        "answer_text": None if unparsed else "option",
        "compatible_theories": None if unparsed else compatible,
        "edt_leaning": None if unparsed else edt_leaning,
        "prosocial_option": None if unparsed or prosocial is None else 0,
        "chose_prosocial": None if unparsed else prosocial,
        "truncated_thinking": False,
        "completion": SENTINEL_COMPLETION,
        "visible_text": SENTINEL_COMPLETION,
        "parsed": not unparsed,
    }


def capability_row(*, correct: bool, parsed: bool = True) -> dict[str, Any]:
    """One arithmetic-canary record."""
    return {
        "record": SECTION_CAPABILITIES,
        "item_index": 0,
        "prompt": SENTINEL_COMPLETION,
        "expected": 7,
        "parsed_answer": 7 if parsed else None,
        "correct": correct,
        "parsed": parsed,
        "truncated_thinking": False,
        "completion": SENTINEL_COMPLETION,
        "visible_text": SENTINEL_COMPLETION,
    }


def survey_row(  # noqa: PLR0913 - one record's worth of labels, not a bundle worth naming
    item_id: str,
    *,
    kind: str = SURVEY_ORDERED_CHOICE,
    instrument: str = "placeholder-instrument",
    subscale: str = "placeholder-subscale",
    family: str = "placeholder-family",
    scale_points: int = 5,
    score: float | None = None,
    canonical_index: int | None = None,
    counterpart: str = COUNTERPART_UNSPECIFIED,
    counterpart_pair: str | None = None,
    announced_tag: str | None = None,
    tag: str | None = None,
    option_labels: tuple[str, ...] = (),
    chosen_label: str | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """One self-report record carrying the whole field contract, with placeholder labels only.

    Item text never appears in a tracked file, so every string here is a placeholder: the tracked
    data an item contributes is its id, its instrument, its subscale and its option labels, and the
    reductions under test read nothing else.
    """
    matched = None if announced_tag is None or tag is None else announced_tag == tag
    canonical = canonical_index
    if canonical is None and score is not None:
        canonical = int(score) - 1
    if kind == SURVEY_CHEAP_TALK and matched is None:
        canonical = None
    return {
        "record": SECTION_SELF_REPORT,
        "item_id": item_id,
        "family": family,
        "instrument": instrument,
        "subscale": subscale,
        "kind": kind,
        "tier": "core",
        "wording": WORDING_AS_PUBLISHED,
        "twin_of": None,
        "counterpart": counterpart,
        "counterpart_pair": counterpart_pair,
        "reverse_keyed": False,
        "scale_points": scale_points,
        "predicts_game": None,
        "option_labels": list(option_labels),
        "parsed": canonical is not None,
        "presented_index": canonical,
        "canonical_index": canonical,
        "response": None if canonical is None else canonical + 1,
        "score": score,
        "numeric": None,
        "tag": tag,
        "announced_tag": announced_tag,
        "statement_matched_action": matched,
        "chosen_label": chosen_label,
        "payoff_self": None,
        "payoff_other": None,
        "orientation": None,
        "sample_index": sample_index,
        "option_order_name": "as-authored",
        "option_order": [],
        "numeric_example": None,
        "truncated_thinking": False,
        "completion": SENTINEL_COMPLETION,
        "visible_text": SENTINEL_COMPLETION,
    }


def cheap_talk_row(
    item_id: str, *, announced: str | None, acted: str | None, sample_index: int = 0
) -> dict[str, Any]:
    """One cheap-talk record: what it announced, and what it then did. Placeholder menu words."""
    return survey_row(
        item_id,
        kind=SURVEY_CHEAP_TALK,
        scale_points=2,
        canonical_index=0 if acted is not None else None,
        announced_tag=announced,
        tag=acted,
        sample_index=sample_index,
    )


def labelled_choice_row(  # noqa: PLR0913 - one record's worth of labels, not a bundle worth naming
    item_id: str,
    *,
    labels: tuple[str, ...],
    chosen: str | None,
    family: str = FAMILY_VALUES_FORCED_CHOICE,
    subscale: str = "placeholder-pole-pair",
    sample_index: int = 0,
) -> dict[str, Any]:
    """One nominal-choice record whose options carry category labels. Placeholder value poles only.

    `chosen=None` means the render did not parse, which nulls the canonical index with it: a parse
    failure is not a vote against every label on the item, so it belongs in neither denominator.
    """
    return survey_row(
        item_id,
        kind=SURVEY_CHOICE,
        family=family,
        subscale=subscale,
        scale_points=len(labels),
        canonical_index=None if chosen is None else labels.index(chosen),
        option_labels=labels,
        chosen_label=chosen,
        sample_index=sample_index,
    )


def control_choice_row(
    item_id: str, *, canonical_index: int | None, sample_index: int = 0
) -> dict[str, Any]:
    """One negative-control nominal-choice record: no labels, so its options are only numbers."""
    return survey_row(
        item_id,
        kind=SURVEY_CHOICE,
        family=FAMILY_NEGATIVE_CONTROL,
        subscale="inert-preference",
        scale_points=3,
        canonical_index=canonical_index,
        sample_index=sample_index,
    )


def default_rows() -> list[dict[str, Any]]:
    """A cell's worth of records touching every section and every column role."""
    return [
        game_row(TWIN_GAME, "twin-1"),
        game_row(TWIN_GAME, "twin-2", coop=0.0),
        game_row(TWIN_GAME, "twin-3", reskin_id="frame-two", payoff_variant="temptation-10"),
        game_row(
            TWIN_GAME, "twin-4", reskin_id="frame-two", payoff_variant="temptation-10", coop=0.0
        ),
        game_row(STAG_GAME, "stag-1", payoff_variant=PINNED_RUNG),
        game_row(STAG_GAME, "stag-2", payoff_variant=OTHER_RUNG, coop=0.0),
        game_row(NEGATIVE_CONTROL, "neg-1", payoff_variant="standard"),
        game_row(NEGATIVE_CONTROL, "neg-2", payoff_variant="standard", coop=0.0),
        iterated_row("iter-1", ["C", "C", "D"]),
        dictator_row("dict-1", keep=0.5),
        dictator_row("dict-2", keep=0.75),
        open_ended_probe("open-1", "CDT"),
        open_ended_probe("open-2", "EDT"),
        choice_probe("choice-1", answer_index=0, order_name="canonical", edt_leaning=1.0),
        choice_probe("choice-1", answer_index=0, order_name="reversed", edt_leaning=1.0),
        capability_row(correct=True),
        capability_row(correct=True),
        capability_row(correct=True),
        capability_row(correct=False),
    ]


def write_cell(  # noqa: PLR0913 - every knob here is one way a cell can be malformed
    battery_dir: Path,
    arm: str,
    step: int,
    *,
    eval_run: str = "run-one",
    rows: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
    trailing_garbage: bool = False,
    drop_meta: bool = False,
) -> Path:
    """Write one synthetic cell where the readout will find it, and return its path."""
    cell_dir = battery_dir / eval_run / arm
    cell_dir.mkdir(parents=True, exist_ok=True)
    path = cell_dir / f"step-{step}.jsonl"
    records: list[dict[str, Any]] = [] if drop_meta else [meta or meta_record(arm, step)]
    records.extend(default_rows() if rows is None else rows)
    text = "".join(f"{json.dumps(record)}\n" for record in records)
    if trailing_garbage:
        text += '{"record": "game-behavior", "game_id": "twin'
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def complete_battery(tmp_path: Path) -> Path:
    """Two arms that both reached both steps of the ladder, with nothing else wrong."""
    battery = tmp_path / "battery-testsha"
    for arm in (GROUP_ARM, SELF_ARM):
        for step in (0, 10):
            write_cell(battery, arm, step)
    return battery


def render(battery_dir: Path) -> str:
    """Build and render one battery, the way the CLI does."""
    readout = battery_tables.build_readout(battery_dir)
    return battery_readout.render_markdown(readout, command="regenerate-me")


class TestReadingCells:
    def test_a_cell_without_a_meta_record_raises(self, tmp_path: Path):
        battery = tmp_path / "battery-nometa"
        write_cell(battery, GROUP_ARM, 0, drop_meta=True)
        with pytest.raises(ValueError, match=RECORD_META):
            battery_tables.build_readout(battery)

    def test_a_meta_record_missing_the_arm_raises(self, tmp_path: Path):
        battery = tmp_path / "battery-noarm"
        meta = meta_record(GROUP_ARM, 0)
        del meta["arm"]
        write_cell(battery, GROUP_ARM, 0, meta=meta)
        with pytest.raises(ValueError, match="arm"):
            battery_tables.build_readout(battery)

    def test_a_battery_with_no_cells_raises(self, tmp_path: Path):
        empty = tmp_path / "battery-empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="Nothing to report on"):
            battery_tables.build_readout(empty)

    def test_a_half_written_cell_is_excluded_rather_than_fatal(self, complete_battery: Path):
        write_cell(complete_battery, RUNG_ARM, 0, trailing_garbage=True)
        readout = battery_tables.build_readout(complete_battery)
        assert [cell.path.name for cell in readout.excluded] == ["step-0.jsonl"]
        assert "unparseable JSON" in readout.excluded[0].reason
        assert RUNG_ARM not in {arm.arm for arm in readout.arms}
        assert {arm.arm for arm in readout.arms} == {GROUP_ARM, SELF_ARM}

    def test_a_duplicate_cell_keeps_the_one_written_later(self, tmp_path: Path):
        battery = tmp_path / "battery-dup"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            eval_run="first-run",
            meta=meta_record(GROUP_ARM, 0, written_at="2026-08-20T01:00:00+00:00"),
        )
        write_cell(
            battery,
            GROUP_ARM,
            0,
            eval_run="second-run",
            meta=meta_record(GROUP_ARM, 0, written_at="2026-08-20T09:00:00+00:00"),
            rows=[game_row(TWIN_GAME, "twin-1", coop=0.0)],
        )
        readout = battery_tables.build_readout(battery)
        assert readout.cell_count == 1
        assert [cell.path.parent.name for cell in readout.excluded] == [GROUP_ARM]
        assert "first-run" in str(readout.excluded[0].path)
        assert "duplicate cell" in readout.excluded[0].reason

    def test_generated_text_never_reaches_the_document(self, complete_battery: Path):
        assert SENTINEL_COMPLETION not in render(complete_battery)


class TestCrossLegPooling:
    """The deliberated and non-deliberated passes are different policies and must never pool.

    Both legs of one arm carry the same `arm` value in their meta -- the `-think` suffix that keeps
    them apart is on the output directory, which nothing here is allowed to read a label from -- so
    they collide on (arm, step) and, before `refuse_pooled_legs`, the deduplication reaped one whole
    leg with the reason "duplicate cell". A reader would have seen a re-synced pass being tidied up,
    not half the experiment leaving.
    """

    def test_a_thinking_off_cell_in_a_thinking_on_arm_directory_is_refused(self, tmp_path: Path):
        """The named sabotage from the battery's validation plan, gate 7."""
        battery = tmp_path / "battery-crossleg"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            eval_run="deliberated",
            meta=meta_record(GROUP_ARM, 0, thinking=True, sampler_mode="training-distribution"),
        )
        write_cell(
            battery,
            GROUP_ARM,
            0,
            eval_run="non-deliberated",
            meta=meta_record(GROUP_ARM, 0, thinking=False, sampler_mode="training-distribution"),
        )
        with pytest.raises(ValueError, match="2 different policies") as raised:
            battery_tables.build_readout(battery)
        message = str(raised.value)
        assert "thinking=True" in message
        assert "thinking=False" in message
        # Refused before the deduplication runs, so no reader is told a leg was a redundant pass.
        assert "duplicate cell" not in message

    def test_two_sampler_modes_under_one_battery_are_refused(self, tmp_path: Path):
        """The other axis of the same gate: one decoding distribution per document."""
        battery = tmp_path / "battery-twosamplers"
        for eval_run, mode in (("traintemp", "training-distribution"), ("vendor", "deployment")):
            write_cell(
                battery,
                GROUP_ARM,
                0,
                eval_run=eval_run,
                meta=meta_record(GROUP_ARM, 0, sampler_mode=mode),
            )
        with pytest.raises(ValueError, match="2 different policies") as raised:
            battery_tables.build_readout(battery)
        assert "sampler_mode=deployment" in str(raised.value)

    def test_legs_split_across_steps_are_refused_too(self, tmp_path: Path):
        """No (arm, step) collision here, and pooling is still wrong.

        `battery_tables` differences an arm's base cell against its pooled late window, so a
        thinking-off step 0 under a thinking-on ladder yields a delta that reads as a training
        effect and is a deliberation effect. The refusal spans the battery, not just the collisions.
        """
        battery = tmp_path / "battery-mixedladder"
        write_cell(battery, GROUP_ARM, 0, meta=meta_record(GROUP_ARM, 0, thinking=False))
        write_cell(battery, GROUP_ARM, 10, meta=meta_record(GROUP_ARM, 10, thinking=True))
        with pytest.raises(ValueError, match="2 different policies"):
            battery_tables.build_readout(battery)

    def test_an_unrecorded_field_refuses_against_a_recorded_one(self, tmp_path: Path):
        """Cells written before a field existed cannot be shown to match cells that name it."""
        battery = tmp_path / "battery-halfrecorded"
        write_cell(battery, GROUP_ARM, 0, eval_run="older", meta=meta_record(GROUP_ARM, 0))
        write_cell(
            battery,
            SELF_ARM,
            0,
            eval_run="newer",
            meta=meta_record(SELF_ARM, 0, sampler_mode="training-distribution"),
        )
        with pytest.raises(ValueError, match="2 different policies") as raised:
            battery_tables.build_readout(battery)
        assert f"sampler_mode={battery_cells.LEG_FIELD_UNRECORDED}" in str(raised.value)

    def test_a_battery_agreeing_on_every_leg_field_reads_normally(self, tmp_path: Path):
        """The half that keeps this from being a gate that can only go red.

        Cells that agree pass, including a battery where nothing records a sampler mode at all --
        which is every cell written before that field existed, so the check cannot retroactively
        refuse the batteries already on disk.
        """
        battery = tmp_path / "battery-oneleg"
        for arm in (GROUP_ARM, SELF_ARM):
            for step in (0, 10):
                write_cell(battery, arm, step, meta=meta_record(arm, step, thinking=True))
        readout = battery_tables.build_readout(battery)
        assert readout.cell_count == 4
        assert not readout.excluded

    def test_a_genuine_same_leg_relaunch_still_deduplicates(self, tmp_path: Path):
        """One leg, one cell written twice, is the case the deduplication is for; it keeps working."""
        battery = tmp_path / "battery-relaunch"
        for eval_run, written_at in (
            ("first-run", "2026-08-22T01:00:00+00:00"),
            ("second-run", "2026-08-22T09:00:00+00:00"),
        ):
            write_cell(
                battery,
                GROUP_ARM,
                0,
                eval_run=eval_run,
                meta=meta_record(
                    GROUP_ARM,
                    0,
                    thinking=True,
                    sampler_mode="training-distribution",
                    written_at=written_at,
                ),
            )
        readout = battery_tables.build_readout(battery)
        assert readout.cell_count == 1
        assert "duplicate cell" in readout.excluded[0].reason
        # Both cells record every leg field, so the exclusion needs no caveat about which is which.
        assert "neither cell records" not in readout.excluded[0].reason

    def test_a_duplicate_whose_leg_was_never_recorded_names_the_missing_field(self, tmp_path: Path):
        """The blind spot the refusal cannot close, surfaced where the exclusion is named.

        Two passes written before `sampler_mode` reached the meta record agree on it vacuously, so
        they collide on (arm, step) and one is kept. That is the live shape of
        `artifacts/games/evals/remote-g7e-20260819/`: a training-sampler pass and a vendor-preset pass
        over the same arms, 24 cells deep, indistinguishable from a relaunch after the fact.
        """
        battery = tmp_path / "battery-unrecordedleg"
        for eval_run, written_at in (
            ("traintemp", "2026-08-19T01:00:00+00:00"),
            ("vendor", "2026-08-19T09:00:00+00:00"),
        ):
            meta = meta_record(GROUP_ARM, 0, written_at=written_at)
            assert "sampler_mode" not in meta
            write_cell(battery, GROUP_ARM, 0, eval_run=eval_run, meta=meta)
        readout = battery_tables.build_readout(battery)
        assert readout.cell_count == 1
        reason = readout.excluded[0].reason
        assert "neither cell records sampler_mode" in reason
        assert "chose one on your behalf" in reason


class TestLadderAndCompleteness:
    def test_the_ladder_is_the_union_of_steps_the_arms_reached(self, tmp_path: Path):
        battery = tmp_path / "battery-ladder"
        write_cell(battery, GROUP_ARM, 0)
        write_cell(battery, GROUP_ARM, 10)
        write_cell(battery, SELF_ARM, 0)
        readout = battery_tables.build_readout(battery)
        assert readout.ladder == (0, 10)
        by_arm = {arm.arm: arm for arm in readout.arms}
        assert by_arm[SELF_ARM].missing_steps == (10,)
        assert by_arm[GROUP_ARM].missing_steps == ()

    def test_an_arm_short_of_the_ladder_banners_incomplete(self, tmp_path: Path):
        battery = tmp_path / "battery-incomplete"
        write_cell(battery, GROUP_ARM, 0)
        write_cell(battery, GROUP_ARM, 10)
        write_cell(battery, SELF_ARM, 0)
        document = render(battery)
        assert "INCOMPLETE AND ERROR-CONTAINING" in document
        assert f"- INCOMPLETE arm `{SELF_ARM}`: 1 of 2 cells" in document
        assert "missing 10" in document
        assert f"## Arm `{SELF_ARM}` -- INCOMPLETE" in document

    def test_a_complete_battery_does_not_banner_incomplete(self, complete_battery: Path):
        document = render(complete_battery)
        assert "INCOMPLETE" not in document
        assert "Every arm reached all 2 steps of the ladder" in document

    @pytest.mark.parametrize(
        ("steps", "expected"),
        [
            ((0, 10, 20, 30, 40, 50, 60, 70), (40, 50, 60, 70)),
            ((0, 10, 20), (10, 20)),
            ((0, 10), (10,)),
            ((0,), ()),
            ((), ()),
        ],
    )
    def test_the_late_window_is_the_upper_half_of_the_ladder(
        self, steps: tuple[int, ...], expected: tuple[int, ...]
    ):
        assert battery_tables.late_window(steps) == expected


class TestDenominators:
    def test_a_trained_game_cell_carries_its_parsed_and_asked_counts(self, tmp_path: Path):
        battery = tmp_path / "battery-kn"
        rows = [
            game_row(TWIN_GAME, "twin-1", coop=1.0),
            game_row(TWIN_GAME, "twin-2", coop=0.0),
            game_row(TWIN_GAME, "twin-3", coop=None),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        readout = battery_tables.build_readout(battery)
        ladder = readout.arms[0].trained_ladder
        assert [row["rate (parsed/asked prompts)"] for row in ladder] == ["0.500 (2/3)"]
        assert [row["parse_failures"] for row in ladder] == [1]
        assert "0.500 (2/3)" in render(battery)

    def test_a_rate_is_a_mean_over_prompts_with_the_draw_counts_beside_it(self, tmp_path: Path):
        """A prompt sampled eight times is one observation, not eight, however many parsed.

        Unequal parsed draw counts are what discriminate the two reductions: where every prompt
        contributes the same number, the pooled mean and the mean of per-prompt means are
        arithmetically identical, so a uniform cell passes under either. Here one prompt answers
        eight times cooperating and its neighbour once defecting, so per prompt the rate is 0.500
        while pooling the draws would read 0.889.
        """
        battery = tmp_path / "battery-draws"
        rows = [
            *(game_row(TWIN_GAME, "twin-loud", coop=1.0) for _ in range(8)),
            game_row(TWIN_GAME, "twin-quiet", coop=0.0),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        ladder = battery_tables.build_readout(battery).arms[0].trained_ladder
        assert ladder[0]["rate (parsed/asked prompts)"] == "0.500 (2/2)"
        assert ladder[0]["draws (parsed/asked)"] == "9/9"
        assert ladder[0]["parse_failures"] == 0

    def test_a_cell_that_parsed_nothing_shows_its_denominator(self, tmp_path: Path):
        battery = tmp_path / "battery-nothing"
        write_cell(battery, GROUP_ARM, 0, rows=[game_row(TWIN_GAME, "twin-1", coop=None)])
        ladder = battery_tables.build_readout(battery).arms[0].trained_ladder
        assert ladder[0]["rate (parsed/asked prompts)"] == "- (0/1)"

    def test_truncated_thinking_is_counted_beside_the_rate(self, tmp_path: Path):
        battery = tmp_path / "battery-truncated"
        rows = [
            game_row(TWIN_GAME, "twin-1"),
            game_row(TWIN_GAME, "twin-2", coop=None, truncated=True),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        arm = battery_tables.build_readout(battery).arms[0]
        assert arm.trained_ladder[0]["n_truncated"] == 1
        health = {row["section"]: row for row in arm.section_health}
        assert health[SECTION_GAME_BEHAVIOR]["truncated/asked"] == "1/2"
        assert health[SECTION_GAME_BEHAVIOR]["parsed/asked"] == "1/2"

    def test_the_canary_reports_both_defensible_denominators(self, tmp_path: Path):
        battery = tmp_path / "battery-canary"
        rows = [
            capability_row(correct=True),
            capability_row(correct=True),
            capability_row(correct=False),
            capability_row(correct=False, parsed=False),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        row = battery_tables.build_readout(battery).arms[0].capability_ladder[0]
        assert row["accuracy over asked (correct/asked)"] == "0.500 (2/4)"
        assert row["accuracy over parsed (correct/parsed)"] == "0.667 (2/3)"


class TestTransferAndControls:
    def test_every_game_gets_a_row_with_its_role(self, complete_battery: Path):
        arm = next(
            entry
            for entry in battery_tables.build_readout(complete_battery).arms
            if entry.arm == GROUP_ARM
        )
        roles = {row["game"]: row["role"] for row in arm.transfer_matrix}
        assert roles[TWIN_GAME] == battery_tables.ROLE_TRAINED
        assert roles[STAG_GAME] == battery_tables.ROLE_TRANSFER
        assert roles[NEGATIVE_CONTROL] == battery_tables.ROLE_NEGATIVE_CONTROL
        assert arm.transfer_matrix[0]["game"] == TWIN_GAME

    def test_the_transfer_matrix_has_one_column_per_step(self, complete_battery: Path):
        arm = battery_tables.build_readout(complete_battery).arms[0]
        assert "step 0" in arm.transfer_matrix[0]
        assert "step 10" in arm.transfer_matrix[0]

    def test_the_pooled_transfer_table_carries_one_move_per_game(self, tmp_path: Path):
        battery = tmp_path / "battery-transferwindow"
        write_cell(battery, GROUP_ARM, 0, rows=[game_row(STAG_GAME, "stag-1", coop=1.0)])
        write_cell(battery, GROUP_ARM, 10, rows=[game_row(STAG_GAME, "stag-1", coop=0.0)])
        arm = battery_tables.build_readout(battery).arms[0]
        row = next(row for row in arm.transfer_windows if row["game"] == STAG_GAME)
        assert row["role"] == battery_tables.ROLE_TRANSFER
        assert row["base step 0 (parsed/asked prompts)"] == "1.000 (1/1)"
        assert row["late steps 10 (parsed/asked prompts)"] == "0.000 (1/1)"
        assert row["delta"] == "-1.000"

    def test_a_game_that_measured_nothing_has_no_move(self, tmp_path: Path):
        battery = tmp_path / "battery-nomove"
        write_cell(battery, GROUP_ARM, 0, rows=[game_row(STAG_GAME, "stag-1", coop=None)])
        write_cell(battery, GROUP_ARM, 10, rows=[game_row(STAG_GAME, "stag-1", coop=0.0)])
        arm = battery_tables.build_readout(battery).arms[0]
        row = next(row for row in arm.transfer_windows if row["game"] == STAG_GAME)
        assert row["base step 0 (parsed/asked prompts)"] == "- (0/1)"
        assert row["delta"] == battery_tables.EMPTY_CELL

    def test_the_negative_control_gets_its_own_ladder(self, complete_battery: Path):
        arm = battery_tables.build_readout(complete_battery).arms[0]
        assert [row["step"] for row in arm.negative_control] == [0, 10]
        assert arm.negative_control[0]["coop-label rate (parsed/asked prompts)"] == "0.500 (2/2)"
        assert NEGATIVE_CONTROL in render(complete_battery)

    def test_the_dictator_measure_is_the_keep_fraction_with_a_note(self, tmp_path: Path):
        battery = tmp_path / "battery-dictator"
        write_cell(battery, DICTATOR_ARM, 0)
        arm = battery_tables.build_readout(battery).arms[0]
        assert arm.trained_ladder[0]["measure"] == battery_tables.KEEP_FIELD
        assert arm.trained_ladder[0]["rate (parsed/asked prompts)"] == "0.625 (2/2)"
        assert any("no cooperation rate at all" in note for note in arm.notes)
        assert "recomputed from the training parquet" in render(battery)

    def test_a_pinned_arm_separates_its_own_rung_from_the_others(self, tmp_path: Path):
        battery = tmp_path / "battery-rungs"
        write_cell(battery, RUNG_ARM, 0)
        arm = battery_tables.build_readout(battery).arms[0]
        roles = {row["payoff_variant"]: row["role"] for row in arm.trained_ladder}
        assert roles[PINNED_RUNG] == "trained variant"
        assert roles[OTHER_RUNG] == "transfer variant (same game)"

    def test_an_unpinned_arm_pools_its_variants(self, complete_battery: Path):
        arm = battery_tables.build_readout(complete_battery).arms[0]
        variants = {row["payoff_variant"] for row in arm.trained_ladder}
        assert variants == {battery_tables.ALL_VARIANTS}

    def test_the_iterated_game_gets_a_per_round_profile(self, complete_battery: Path):
        arm = battery_tables.build_readout(complete_battery).arms[0]
        profile = [row for row in arm.per_round if row["step"] == 0]
        assert [row["round_number"] for row in profile] == [1, 2, 3]
        assert [row["coop_rate"] for row in profile] == [1.0, 1.0, 0.0]


class TestSplits:
    def test_the_variant_split_contrasts_the_base_cell_with_the_late_window(self, tmp_path: Path):
        battery = tmp_path / "battery-splits"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                game_row(TWIN_GAME, "twin-1", payoff_variant="temptation-2", coop=1.0),
                game_row(TWIN_GAME, "twin-2", payoff_variant="temptation-10", coop=1.0),
            ],
        )
        write_cell(
            battery,
            GROUP_ARM,
            10,
            rows=[
                game_row(TWIN_GAME, "twin-1", payoff_variant="temptation-2", coop=1.0),
                game_row(TWIN_GAME, "twin-2", payoff_variant="temptation-10", coop=0.0),
            ],
        )
        arm = battery_tables.build_readout(battery).arms[0]
        splits = {row["payoff_variant"]: row for row in arm.variant_splits}
        late_column = "late steps 10 (parsed/asked prompts)"
        assert splits["temptation-2"]["base step 0 (parsed/asked prompts)"] == "1.000 (1/1)"
        assert splits["temptation-2"][late_column] == "1.000 (1/1)"
        assert splits["temptation-2"]["delta"] == "+0.000"
        assert splits["temptation-10"][late_column] == "0.000 (1/1)"
        assert splits["temptation-10"]["delta"] == "-1.000"

    def test_the_reskin_split_names_every_frame(self, complete_battery: Path):
        arm = battery_tables.build_readout(complete_battery).arms[0]
        assert {row["reskin_id"] for row in arm.reskin_splits} == {"frame-one", "frame-two"}
        assert all(row["role"] == "surface frame" for row in arm.reskin_splits)

    def test_a_battery_with_only_a_base_cell_says_its_late_window_is_empty(self, tmp_path: Path):
        battery = tmp_path / "battery-base-only"
        write_cell(battery, GROUP_ARM, 0)
        arm = battery_tables.build_readout(battery).arms[0]
        assert arm.late_window == ()
        assert any("late window is empty" in note for note in arm.notes)

    def test_a_single_cell_late_window_is_named_as_one(self, tmp_path: Path):
        battery = tmp_path / "battery-thin-window"
        write_cell(battery, GROUP_ARM, 0)
        write_cell(battery, GROUP_ARM, 10)
        arm = battery_tables.build_readout(battery).arms[0]
        assert arm.late_window == (10,)
        assert any("single cell (step 10)" in note for note in arm.notes)
        assert "one cell against one cell" in render(battery)

    def test_a_two_cell_late_window_needs_no_single_cell_note(self, tmp_path: Path):
        battery = tmp_path / "battery-wide-window"
        for step in (0, 10, 20):
            write_cell(battery, GROUP_ARM, step)
        arm = battery_tables.build_readout(battery).arms[0]
        assert arm.late_window == (10, 20)
        assert not any("single cell" in note for note in arm.notes)


class TestDecisionTheory:
    def test_the_ladder_carries_a_column_per_theory_label(self, complete_battery: Path):
        row = battery_tables.build_readout(complete_battery).arms[0].dt_ladder[0]
        assert row["n_CDT"] == 1
        assert row["n_EDT"] == 1
        assert row["n_FDT"] == 0
        assert row["n_ambiguous"] == 0
        assert row["open-ended (parsed/asked)"] == "2/2"

    def test_the_scalar_measures_carry_their_used_and_asked_denominators(
        self, complete_battery: Path
    ):
        row = battery_tables.build_readout(complete_battery).arms[0].dt_ladder[0]
        assert row["mean_edt_leaning"] == "1.000 (1/1 items)"
        assert row["prosocial_choice_rate"] == "1.000 (1/1 valenced items)"
        assert row["order_disagreement_rate"] == "0.000 (1/1 pairs)"

    def test_an_unparsed_render_moves_the_used_side_but_not_the_asked_side(self, tmp_path: Path):
        battery = tmp_path / "battery-floating"
        rows = [
            choice_probe("choice-1", answer_index=0, order_name="canonical", edt_leaning=1.0),
            choice_probe("choice-1", answer_index=0, order_name="reversed", edt_leaning=1.0),
            choice_probe("choice-2", answer_index=None, order_name="canonical", edt_leaning=None),
            choice_probe("choice-2", answer_index=None, order_name="reversed", edt_leaning=None),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        row = battery_tables.build_readout(battery).arms[0].dt_ladder[0]
        assert row["mean_edt_leaning"] == "1.000 (1/2 items)"
        assert row["prosocial_choice_rate"] == "1.000 (1/1 valenced items)"
        assert row["order_disagreement_rate"] == "0.000 (1/2 pairs)"

    def test_an_order_sensitive_item_scores_disagreement(self, tmp_path: Path):
        battery = tmp_path / "battery-order"
        rows = [
            choice_probe("choice-1", answer_index=0, order_name="canonical", edt_leaning=1.0),
            choice_probe("choice-1", answer_index=1, order_name="reversed", edt_leaning=0.0),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        row = battery_tables.build_readout(battery).arms[0].dt_ladder[0]
        assert row["order_disagreement_rate"] == "1.000 (1/1 pairs)"

    def test_flips_compare_the_first_and_last_landed_cell(self, tmp_path: Path):
        battery = tmp_path / "battery-flips"
        write_cell(battery, GROUP_ARM, 0, rows=[open_ended_probe("open-1", "CDT")])
        write_cell(battery, GROUP_ARM, 10, rows=[open_ended_probe("open-1", "EDT")])
        arm = battery_tables.build_readout(battery).arms[0]
        assert arm.dt_flips == [
            {
                "arm": GROUP_ARM,
                "probe_id": "open-1",
                "from_step": 0,
                "to_step": 10,
                "before": "CDT",
                "after": "EDT",
            }
        ]

    def test_the_flip_prose_carries_the_comparable_denominator(self, tmp_path: Path):
        battery = tmp_path / "battery-flipscope"
        rows = [open_ended_probe("open-1", "CDT"), open_ended_probe("open-2", "CDT")]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        write_cell(battery, GROUP_ARM, 10, rows=[open_ended_probe("open-1", "EDT"), *rows[1:]])
        readout = battery_tables.build_readout(battery)
        assert readout.arms[0].flip_scope == battery_tables.FlipScope(
            n_asked_both=2, n_comparable=2
        )
        document = battery_readout.render_markdown(readout, command="regenerate-me")
        assert "1 item(s) changed endorsement, out of 2 comparable at both endpoints" in document
        assert "0 of the 2 items asked at both endpoints were set aside" in document

    def test_an_order_disagreeing_item_leaves_the_comparable_denominator(self, tmp_path: Path):
        battery = tmp_path / "battery-flipsetaside"
        disagreeing = [
            choice_probe("choice-1", answer_index=0, order_name="canonical", edt_leaning=1.0),
            choice_probe("choice-1", answer_index=1, order_name="reversed", edt_leaning=0.0),
        ]
        agreeing = [
            choice_probe("choice-1", answer_index=0, order_name="canonical", edt_leaning=1.0),
            choice_probe("choice-1", answer_index=0, order_name="reversed", edt_leaning=1.0),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=disagreeing)
        write_cell(battery, GROUP_ARM, 10, rows=agreeing)
        scope = battery_tables.build_readout(battery).arms[0].flip_scope
        assert scope == battery_tables.FlipScope(n_asked_both=1, n_comparable=0)
        assert scope.n_set_aside == 1

    def test_the_churn_floor_pools_neighbouring_checkpoints(self, tmp_path: Path):
        battery = tmp_path / "battery-churn"
        write_cell(battery, GROUP_ARM, 0, rows=[open_ended_probe("open-1", "CDT")])
        write_cell(battery, GROUP_ARM, 10, rows=[open_ended_probe("open-1", "EDT")])
        write_cell(battery, GROUP_ARM, 20, rows=[open_ended_probe("open-1", "EDT")])
        readout = battery_tables.build_readout(battery)
        assert readout.arms[0].churn == battery_tables.ChurnFloor(
            n_flips=1, n_comparable=2, n_step_pairs=2
        )
        document = battery_readout.render_markdown(readout, command="regenerate-me")
        assert (
            "Adjacent-checkpoint churn floor for this statistic: 1/2 item-comparisons changed "
            "endorsement across 2 neighbouring-step pairs = 0.500" in document
        )

    def test_a_two_cell_arm_says_its_churn_floor_separates_nothing(self, tmp_path: Path):
        battery = tmp_path / "battery-churndegenerate"
        write_cell(battery, GROUP_ARM, 0, rows=[open_ended_probe("open-1", "CDT")])
        write_cell(battery, GROUP_ARM, 10, rows=[open_ended_probe("open-1", "EDT")])
        readout = battery_tables.build_readout(battery)
        assert readout.arms[0].churn.n_step_pairs == 1
        document = battery_readout.render_markdown(readout, command="regenerate-me")
        assert "the floor separates nothing yet" in document

    def test_the_endorsement_table_counts_every_pattern_per_step(self, tmp_path: Path):
        battery = tmp_path / "battery-endorse"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                choice_probe(
                    "choice-1",
                    answer_index=0,
                    order_name="canonical",
                    edt_leaning=0.0,
                    compatible=["CDT", "EDT", "FDT"],
                ),
                choice_probe(
                    "choice-2",
                    answer_index=0,
                    order_name="canonical",
                    edt_leaning=0.0,
                    compatible=[],
                ),
                choice_probe(
                    "choice-3", answer_index=None, order_name="canonical", edt_leaning=None
                ),
            ],
        )
        write_cell(
            battery,
            GROUP_ARM,
            10,
            rows=[
                choice_probe(
                    "choice-1",
                    answer_index=0,
                    order_name="canonical",
                    edt_leaning=1.0,
                    compatible=["FDT"],
                )
            ],
        )
        arm = battery_tables.build_readout(battery).arms[0]
        assert arm.dt_endorsements == [
            {
                "step": 0,
                "CDT+EDT+FDT": 1,
                "FDT": 0,
                "neither": 1,
                "unparsed": 1,
                "renders asked": 3,
            },
            {
                "step": 10,
                "CDT+EDT+FDT": 0,
                "FDT": 1,
                "neither": 0,
                "unparsed": 0,
                "renders asked": 1,
            },
        ]
        assert "theory endorsement per multiple-choice render" in render(battery)

    def test_a_theory_label_outside_the_vocabulary_raises(self, tmp_path: Path):
        battery = tmp_path / "battery-badlabel"
        write_cell(battery, GROUP_ARM, 0, rows=[open_ended_probe("open-1", "made-up-theory")])
        with pytest.raises(ValueError, match="made-up-theory"):
            battery_tables.build_readout(battery)

    def test_one_cell_cannot_produce_a_flip_table(self, tmp_path: Path):
        battery = tmp_path / "battery-oneflip"
        write_cell(battery, GROUP_ARM, 0)
        assert battery_tables.build_readout(battery).arms[0].dt_flips == []
        assert "there is nothing to compare yet" in render(battery)

    def test_an_incomplete_arm_says_its_flip_endpoints_are_not_the_end_of_training(
        self, tmp_path: Path
    ):
        battery = tmp_path / "battery-flipbanner"
        write_cell(battery, GROUP_ARM, 0)
        write_cell(battery, GROUP_ARM, 10)
        write_cell(battery, SELF_ARM, 0)
        write_cell(battery, SELF_ARM, 10)
        write_cell(battery, SELF_ARM, 20)
        document = render(battery)
        assert "comparing step 0 against step 10 -- NOT the end of training" in document
        assert "comparing step 0 against step 20." in document


GAIN_ONLY_LADDER, MIXED_LADDER = battery_tables.RISK_LOSS_AVERSION_MIRROR_PAIR

# Placeholder menu words, never a real item's vocabulary: the remote is public.
ANNOUNCE_WORD = "wordnumber1"
OTHER_WORD = "wordnumber2"

COUNTERPART_PAIR = "placeholder-counterpart-pair"
CHEAP_TALK_ITEM = "placeholder-cheap-talk-one"
OTHER_CHEAP_TALK_ITEM = "placeholder-cheap-talk-two"


def gain_only_row(score: float | None, *, scale_points: int = 5) -> dict[str, Any]:
    """One render of the pure-gain gamble ladder, whose score is the rung it chose."""
    return survey_row(
        GAIN_ONLY_LADDER,
        instrument="risk-gamble-ladder",
        subscale="variance-tolerance",
        scale_points=scale_points,
        score=score,
    )


def mixed_row(score: float | None, *, scale_points: int = 5) -> dict[str, Any]:
    """One render of the mirrored gain-or-loss ladder: the same spreads, a negative low branch."""
    return survey_row(
        MIXED_LADDER,
        instrument="loss-gamble-ladder",
        subscale="loss-exposure-tolerance",
        scale_points=scale_points,
        score=score,
    )


def counterpart_row(  # noqa: PLR0913 - one record's worth of labels, not a bundle worth naming
    item_id: str,
    *,
    counterpart: str,
    pair: str = COUNTERPART_PAIR,
    kind: str = SURVEY_ORDERED_CHOICE,
    score: float | None = None,
    canonical_index: int | None = None,
) -> dict[str, Any]:
    """One arm of a counterpart pair: the same item, asked about an AI or about a human."""
    return survey_row(
        item_id,
        kind=kind,
        counterpart=counterpart,
        counterpart_pair=pair,
        score=score,
        canonical_index=canonical_index,
    )


class TestTheRiskMirrorPair:
    def test_the_pair_is_the_two_authored_mirror_items(self):
        """The pair is hard-coded, so its contents are what a test has to pin."""
        assert battery_tables.RISK_LOSS_AVERSION_MIRROR_PAIR == (
            "risk-sure-versus-spread-gain-only-mirror",
            "loss-mixed-versus-sure-mirror",
        )

    def test_the_gap_is_the_gain_only_mean_rung_minus_the_mixed_one(self, tmp_path: Path):
        battery = tmp_path / "battery-riskmirror"
        rows = [
            gain_only_row(4.0),
            gain_only_row(2.0),
            gain_only_row(None),
            mixed_row(2.0),
            mixed_row(1.0),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        mirror = battery_tables.build_readout(battery).arms[0].self_report_risk_mirror
        assert mirror == [
            {
                "step": 0,
                "gain-only mean rung": "3.000",
                "mixed mean rung": "1.500",
                "loss aversion (gain-only - mixed)": "+1.500",
                "gain-only scored/asked": "2/3",
                "mixed scored/asked": "2/2",
            }
        ]
        assert "+1.500" in render(battery)

    def test_a_missing_ladder_holds_the_row_open_and_says_which_arm_is_absent(self, tmp_path: Path):
        battery = tmp_path / "battery-riskmirror-onearm"
        write_cell(battery, GROUP_ARM, 0, rows=[gain_only_row(3.0)])
        row = battery_tables.build_readout(battery).arms[0].self_report_risk_mirror[0]
        assert row["gain-only mean rung"] == "3.000"
        assert row["mixed mean rung"] == battery_tables.EMPTY_CELL
        assert row["loss aversion (gain-only - mixed)"] == (
            f"- ({battery_tables.RISK_MIRROR_LADDER_MISSING})"
        )
        assert row["gain-only scored/asked"] == "1/1"
        assert row["mixed scored/asked"] == "0/0"

    def test_ladders_of_different_lengths_refuse_a_gap_but_still_print_their_rungs(
        self, tmp_path: Path
    ):
        """The one commensurability check these two items get: they are in different subscales.

        `assert_ordered_choice_ladders_are_commensurable` refuses mixed rung counts WITHIN a
        subscale, and this pair spans two, so nothing upstream would catch a five-rung ladder
        differenced against a six-rung one -- it would print as a plausible loss-aversion number in
        no unit at all.
        """
        battery = tmp_path / "battery-riskmirror-lengths"
        rows = [gain_only_row(4.0, scale_points=5), mixed_row(2.0, scale_points=6)]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        row = battery_tables.build_readout(battery).arms[0].self_report_risk_mirror[0]
        assert row["gain-only mean rung"] == "4.000"
        assert row["mixed mean rung"] == "2.000"
        assert row["loss aversion (gain-only - mixed)"] == (
            f"- ({battery_tables.RISK_MIRROR_RUNG_COUNTS_DIFFER}: [5, 6])"
        )

    def test_a_ladder_that_scored_nothing_reports_its_denominator_and_no_gap(self, tmp_path: Path):
        battery = tmp_path / "battery-riskmirror-unparsed"
        write_cell(battery, GROUP_ARM, 0, rows=[gain_only_row(None), mixed_row(2.0)])
        row = battery_tables.build_readout(battery).arms[0].self_report_risk_mirror[0]
        assert row["gain-only mean rung"] == battery_tables.EMPTY_CELL
        assert row["gain-only scored/asked"] == "0/1"
        assert row["loss aversion (gain-only - mixed)"] == (
            f"- ({battery_tables.RISK_MIRROR_NOTHING_PARSED})"
        )

    def test_a_step_that_administered_neither_ladder_gets_no_row(self, tmp_path: Path):
        battery = tmp_path / "battery-riskmirror-absent"
        write_cell(battery, GROUP_ARM, 0, rows=[gain_only_row(3.0), mixed_row(2.0)])
        write_cell(battery, GROUP_ARM, 10, rows=[survey_row("placeholder-other-item", score=3.0)])
        mirror = battery_tables.build_readout(battery).arms[0].self_report_risk_mirror
        assert [row["step"] for row in mirror] == [0]

    def test_the_prose_names_both_items_and_calls_no_direction_on_the_movement(
        self, complete_battery: Path
    ):
        """The registered expectation is a positive gap at step 0 and nothing about its movement."""
        document = render(complete_battery)
        assert f"`{GAIN_ONLY_LADDER}` against" in document
        assert f"`{MIXED_LADDER}`" in document
        assert "the gap is POSITIVE at step 0, and no direction is called on how it moves" in (
            document
        )
        assert "within-family placebo" in document


class TestCounterpartGaps:
    def test_the_gap_is_the_ai_arm_minus_the_human_one(self, tmp_path: Path):
        battery = tmp_path / "battery-counterpart"
        rows = [
            counterpart_row("placeholder-ai-arm", counterpart=COUNTERPART_AI, score=4.0),
            counterpart_row("placeholder-ai-arm", counterpart=COUNTERPART_AI, score=2.0),
            counterpart_row("placeholder-human-arm", counterpart=COUNTERPART_HUMAN, score=1.0),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        counterparts = battery_tables.build_readout(battery).arms[0].self_report_counterparts
        assert counterparts == [
            {
                "step": 0,
                "counterpart pair": COUNTERPART_PAIR,
                "ai counterpart": "3.000",
                "human counterpart": "1.000",
                "gap (ai - human)": "+2.000",
                "ai parsed/asked": "2/2",
                "human parsed/asked": "1/1",
            }
        ]
        assert "+2.000" in render(battery)

    def test_a_pair_missing_an_arm_prints_its_reason_and_both_denominators(self, tmp_path: Path):
        battery = tmp_path / "battery-counterpart-onearm"
        rows = [counterpart_row("placeholder-ai-arm", counterpart=COUNTERPART_AI, score=4.0)]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        row = battery_tables.build_readout(battery).arms[0].self_report_counterparts[0]
        assert row["ai counterpart"] == battery_tables.EMPTY_CELL
        assert row["human counterpart"] == battery_tables.EMPTY_CELL
        assert row["gap (ai - human)"] == f"- ({COUNTERPART_ARM_MISSING})"
        assert row["ai parsed/asked"] == "1/1"
        assert row["human parsed/asked"] == "0/0"

    def test_a_pair_where_nothing_parsed_reads_differently_from_a_nominal_one(self, tmp_path: Path):
        """Two reasons a pair has no difference; only one of them is a problem with the run."""
        battery = tmp_path / "battery-counterpart-reasons"
        unparsed = [
            counterpart_row(
                "placeholder-ai-arm",
                counterpart=COUNTERPART_AI,
                pair="placeholder-unparsed-pair",
                score=None,
            ),
            counterpart_row(
                "placeholder-human-arm",
                counterpart=COUNTERPART_HUMAN,
                pair="placeholder-unparsed-pair",
                score=3.0,
            ),
        ]
        nominal = [
            counterpart_row(
                "placeholder-ai-nominal",
                counterpart=COUNTERPART_AI,
                pair="placeholder-nominal-pair",
                kind=SURVEY_CHOICE,
                canonical_index=1,
            ),
            counterpart_row(
                "placeholder-human-nominal",
                counterpart=COUNTERPART_HUMAN,
                pair="placeholder-nominal-pair",
                kind=SURVEY_CHOICE,
                canonical_index=0,
            ),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=[*unparsed, *nominal])
        by_pair = {
            row["counterpart pair"]: row
            for row in battery_tables.build_readout(battery).arms[0].self_report_counterparts
        }
        assert by_pair["placeholder-unparsed-pair"]["gap (ai - human)"] == (
            f"- ({COUNTERPART_NOTHING_PARSED})"
        )
        assert by_pair["placeholder-unparsed-pair"]["ai parsed/asked"] == "0/1"
        assert by_pair["placeholder-unparsed-pair"]["human parsed/asked"] == "1/1"
        assert by_pair["placeholder-nominal-pair"]["gap (ai - human)"] == (
            f"- ({COUNTERPART_NOMINAL_ANSWER})"
        )

    def test_the_prose_says_the_difference_is_the_quantity(self, complete_battery: Path):
        document = render(complete_battery)
        assert "counterpart gaps, an AI counterpart against a human one" in document
        assert "One arm's level is not a reading" in document


class TestCheapTalk:
    def test_the_match_rate_carries_its_denominators_and_the_announced_only_count(
        self, tmp_path: Path
    ):
        battery = tmp_path / "battery-cheaptalk"
        rows = [
            cheap_talk_row(CHEAP_TALK_ITEM, announced=ANNOUNCE_WORD, acted=ANNOUNCE_WORD),
            cheap_talk_row(
                CHEAP_TALK_ITEM, announced=ANNOUNCE_WORD, acted=ANNOUNCE_WORD, sample_index=1
            ),
            cheap_talk_row(
                CHEAP_TALK_ITEM, announced=ANNOUNCE_WORD, acted=OTHER_WORD, sample_index=2
            ),
            cheap_talk_row(CHEAP_TALK_ITEM, announced=ANNOUNCE_WORD, acted=None, sample_index=3),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        cheap_talk = battery_tables.build_readout(battery).arms[0].self_report_cheap_talk
        assert cheap_talk == [
            {
                "step": 0,
                "item": CHEAP_TALK_ITEM,
                "match rate": "0.667",
                "parsed/asked": "3/4",
                "announced only": "1",
                "announced -> acted": (
                    f"{ANNOUNCE_WORD} -> {ANNOUNCE_WORD}: 2, {ANNOUNCE_WORD} -> {OTHER_WORD}: 1"
                ),
            }
        ]

    def test_two_items_are_reported_apart_rather_than_pooled(self, tmp_path: Path):
        """A mean over items describing different situations is not a rate of anything."""
        battery = tmp_path / "battery-cheaptalk-peritem"
        rows = [
            cheap_talk_row(CHEAP_TALK_ITEM, announced=ANNOUNCE_WORD, acted=ANNOUNCE_WORD),
            cheap_talk_row(
                CHEAP_TALK_ITEM, announced=ANNOUNCE_WORD, acted=ANNOUNCE_WORD, sample_index=1
            ),
            cheap_talk_row(OTHER_CHEAP_TALK_ITEM, announced=ANNOUNCE_WORD, acted=OTHER_WORD),
            cheap_talk_row(
                OTHER_CHEAP_TALK_ITEM, announced=ANNOUNCE_WORD, acted=OTHER_WORD, sample_index=1
            ),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        cheap_talk = battery_tables.build_readout(battery).arms[0].self_report_cheap_talk
        by_item = {row["item"]: row for row in cheap_talk}
        assert len(cheap_talk) == 2
        assert by_item[CHEAP_TALK_ITEM]["match rate"] == "1.000"
        assert by_item[OTHER_CHEAP_TALK_ITEM]["match rate"] == "0.000"
        assert all(row["parsed/asked"] == "2/2" for row in cheap_talk)

    def test_an_item_that_never_acted_reports_no_rate_and_says_what_it_did(self, tmp_path: Path):
        battery = tmp_path / "battery-cheaptalk-noaction"
        rows = [
            cheap_talk_row(CHEAP_TALK_ITEM, announced=ANNOUNCE_WORD, acted=None),
            cheap_talk_row(CHEAP_TALK_ITEM, announced=None, acted=None, sample_index=1),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        row = battery_tables.build_readout(battery).arms[0].self_report_cheap_talk[0]
        assert row["match rate"] == battery_tables.EMPTY_CELL
        assert row["parsed/asked"] == "0/2"
        assert row["announced only"] == "1"
        assert row["announced -> acted"] == battery_tables.EMPTY_CELL

    def test_the_prose_says_the_items_are_never_pooled(self, complete_battery: Path):
        document = render(complete_battery)
        assert "cheap talk, announcement against action" in document
        assert "how often it did what it said is not a rate of anything" in document
        assert "a coverage fact about the format rather than dishonesty" in document


# Placeholder menu words for the forced-tag items, never a real item's vocabulary: the remote is
# public. Their alphabetical order (one, three, two) differs from the menu order they are given
# below, which is what lets the ordering assertion mean something.
MENU_WORD_ONE = "wordnumberone"
MENU_WORD_TWO = "wordnumbertwo"
MENU_WORD_THREE = "wordnumberthree"

TAGGED_ITEM = "placeholder-tagged-one"
OTHER_TAGGED_ITEM = "placeholder-tagged-two"


def tagged_row(
    item_id: str,
    *,
    word: str | None,
    position: int = 0,
    menu_words: int = 3,
    sample_index: int = 0,
) -> dict[str, Any]:
    """One forced-tag record: which word of a `menu_words`-wide menu it chose, or none at all."""
    return survey_row(
        item_id,
        kind=SURVEY_TAGGED,
        instrument="placeholder-self-characterisation",
        subscale="placeholder-stance",
        scale_points=menu_words,
        canonical_index=None if word is None else position,
        tag=word,
        sample_index=sample_index,
    )


class TestTaggedItems:
    def test_the_shares_carry_their_denominators_the_menu_width_and_the_headroom(
        self, tmp_path: Path
    ):
        """Four renders: the first word twice, the second once, one that parsed to nothing.

        Shares are over the three PARSED renders (0.67 and 0.33) with `3/4` printed beside them, so a
        distribution cannot be read without the parse failures it excludes. Entropy of (2/3, 1/3) is
        0.918 bits against the 1.585 a flat three-word menu would carry, and `menu words` is what
        makes a share legible at all: 0.33 is the flat answer on three words and a peak on six.
        """
        battery = tmp_path / "battery-tagged"
        rows = [
            tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE),
            tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE, sample_index=1),
            tagged_row(TAGGED_ITEM, word=MENU_WORD_TWO, position=1, sample_index=2),
            tagged_row(TAGGED_ITEM, word=None, sample_index=3),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        tagged = battery_tables.build_readout(battery).arms[0].self_report_tagged
        assert tagged == [
            {
                "step": 0,
                "item": TAGGED_ITEM,
                "subscale": "placeholder-stance",
                "menu words": "3",
                "shares": f"{MENU_WORD_ONE}:0.67, {MENU_WORD_TWO}:0.33",
                "entropy (bits)": "0.918",
                "parsed/asked": "3/4",
                "TV vs step0": "0.000",
            }
        ]

    def test_the_words_print_in_menu_order_rather_than_by_name(self, tmp_path: Path):
        """Menu order is one, two, three; alphabetical order is one, three, two. Two of these menus
        are ladders, so a column order that moves with the shares reads down no ladder."""
        battery = tmp_path / "battery-tagged-order"
        rows = [
            tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE, position=0),
            tagged_row(TAGGED_ITEM, word=MENU_WORD_TWO, position=1, sample_index=1),
            tagged_row(TAGGED_ITEM, word=MENU_WORD_THREE, position=2, sample_index=2),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        row = battery_tables.build_readout(battery).arms[0].self_report_tagged[0]
        assert row["shares"] == (
            f"{MENU_WORD_ONE}:0.33, {MENU_WORD_TWO}:0.33, {MENU_WORD_THREE}:0.33"
        )

    def test_the_movement_is_total_variation_distance_from_step_zero(self, tmp_path: Path):
        """Step 0 answers the first word every time and step 10 the second, so the distributions are
        disjoint and the distance is the largest the measure returns."""
        battery = tmp_path / "battery-tagged-movement"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE),
                tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE, sample_index=1),
            ],
        )
        write_cell(
            battery,
            GROUP_ARM,
            10,
            rows=[
                tagged_row(TAGGED_ITEM, word=MENU_WORD_TWO, position=1),
                tagged_row(TAGGED_ITEM, word=MENU_WORD_TWO, position=1, sample_index=1),
            ],
        )
        by_step = {
            row["step"]: row
            for row in battery_tables.build_readout(battery).arms[0].self_report_tagged
        }
        assert by_step[0]["TV vs step0"] == "0.000"
        assert by_step[10]["TV vs step0"] == "1.000"
        assert by_step[10]["shares"] == f"{MENU_WORD_TWO}:1.00"

    def test_a_re_authored_menu_prints_its_reason_rather_than_a_distance(self, tmp_path: Path):
        """The same item id asked with a three-word menu at step 0 and a two-word one at step 10.

        Computed anyway this pair reads 1.000, the largest movement in the battery, when what changed
        was the question. A dropped menu word and a share that fell to zero look identical in a
        record, so the reason is printed and the number withheld.
        """
        battery = tmp_path / "battery-tagged-reauthored"
        write_cell(
            battery, GROUP_ARM, 0, rows=[tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE, menu_words=3)]
        )
        write_cell(
            battery,
            GROUP_ARM,
            10,
            rows=[tagged_row(TAGGED_ITEM, word=MENU_WORD_TWO, position=1, menu_words=2)],
        )
        by_step = {
            row["step"]: row
            for row in battery_tables.build_readout(battery).arms[0].self_report_tagged
        }
        assert by_step[10]["TV vs step0"] == f"- ({TAGGED_VOCABULARY_WIDTH_DIFFERS})"
        assert by_step[10]["menu words"] == "2"

    def test_an_item_the_base_cell_never_asked_says_so(self, tmp_path: Path):
        """A step that added an item and a step whose answers all moved are different facts, and a
        blank cell would invite reading the first as the second."""
        battery = tmp_path / "battery-tagged-newitem"
        write_cell(battery, GROUP_ARM, 0, rows=[tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE)])
        write_cell(
            battery,
            GROUP_ARM,
            10,
            rows=[
                tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE),
                tagged_row(OTHER_TAGGED_ITEM, word=MENU_WORD_TWO, position=1),
            ],
        )
        rows = battery_tables.build_readout(battery).arms[0].self_report_tagged
        added = next(row for row in rows if row["item"] == OTHER_TAGGED_ITEM)
        assert added["TV vs step0"] == f"- ({TAGGED_ITEM_MISSING})"

    def test_an_item_nothing_parsed_on_reports_no_shares_and_no_headroom(self, tmp_path: Path):
        battery = tmp_path / "battery-tagged-silent"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                tagged_row(TAGGED_ITEM, word=None),
                tagged_row(TAGGED_ITEM, word=None, sample_index=1),
            ],
        )
        row = battery_tables.build_readout(battery).arms[0].self_report_tagged[0]
        assert row["shares"] == battery_tables.EMPTY_CELL
        assert row["entropy (bits)"] == battery_tables.EMPTY_CELL
        assert row["parsed/asked"] == "0/2"
        assert row["TV vs step0"] == f"- ({TAGGED_NOTHING_PARSED})"
        assert row["menu words"] == "3"

    def test_an_item_answered_one_way_reports_zero_headroom_and_not_a_negative_one(
        self, tmp_path: Path
    ):
        """Entropy cannot be negative, and `-0.000` in the headroom column would say it can.

        An item answered with one word in every render has entropy exactly zero, and negating the whole
        sum rather than each term returns the float `-0.0`. This is the column a reader consults to
        decide whether an item had any room to move at all, so a minus sign in it misstates what the
        number means -- and it appears on precisely the items whose flat 0.000 movement the column
        exists to qualify.
        """
        battery = tmp_path / "battery-tagged-zero-headroom"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE),
                tagged_row(TAGGED_ITEM, word=MENU_WORD_ONE, sample_index=1),
            ],
        )
        row = battery_tables.build_readout(battery).arms[0].self_report_tagged[0]
        assert row["entropy (bits)"] == "0.000"

    def test_the_prose_says_the_distribution_is_the_reading(self, complete_battery: Path):
        document = render(complete_battery)
        assert "forced-tag items, the distribution over each item's word menu" in document
        assert "never a mean over menu positions" in document
        assert "a response to the lettered format rather than an attribution" in document


# Value pole names are tracked code rather than item text -- `option_labels` lives in the module, not
# in the gitignored option prose -- so these are the real ones. The item ids stay placeholders, and
# the prefixed pole is spelled off `NON_SOCIAL_LABEL_PREFIX` so a rename of the convention reaches here.
JOINT_GAIN = "joint-gain"
OWN_GAIN = "own-gain"
EQUALITY = "equality"
NON_SOCIAL_BREVITY = f"{NON_SOCIAL_LABEL_PREFIX}brevity"
NON_SOCIAL_HONESTY = f"{NON_SOCIAL_LABEL_PREFIX}honesty"
POOLED_NON_SOCIAL = f"{NON_SOCIAL_LABEL_PREFIX}*"

VALUES_ITEM = "placeholder-values-one"
OTHER_VALUES_ITEM = "placeholder-values-two"
THIRD_VALUES_ITEM = "placeholder-values-three"
GRADED_CHOICE_ITEM = "placeholder-graded-dimension-one"
CONTROL_CHOICE_ITEM = "placeholder-control-one"

# The graded-dimension menu, whose labels must never join a values ordering: it is a different
# question on a six-word menu, and the reduction that reads either one pools by label string.
GRADED_LABEL = "own-payoff"
OTHER_GRADED_LABEL = "joint-payoff"


def values_ordering(battery_dir: Path) -> list[dict[str, Any]]:
    """Build one battery and return its first arm's value-pole ordering rows."""
    return battery_tables.build_readout(battery_dir).arms[0].self_report_values_ordering


class TestValuePoleOrdering:
    def test_the_win_rate_is_the_mean_over_items_with_both_denominators(self, tmp_path: Path):
        """Two items, hand-worked. `values-one` offers joint-gain against own-gain and parses four
        times -- joint, joint, joint, own -- so joint-gain wins 3/4 there, and its fifth render parsed
        nothing and belongs in neither denominator. `values-two` offers joint-gain against equality
        once each, so joint-gain wins 1/2 there.

        joint-gain's rate is the mean over ITEMS: (0.75 + 0.5) / 2 = 0.625. Pooling the renders would
        give 4/6 = 0.667, which is the number this table must not print -- it would weight a pole by
        how many of its items happened to parse. Both denominators travel because 0.625 over two items
        and 0.625 over twenty read identically without them.
        """
        battery = tmp_path / "battery-values"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                labelled_choice_row(
                    VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=JOINT_GAIN, sample_index=0
                ),
                labelled_choice_row(
                    VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=JOINT_GAIN, sample_index=1
                ),
                labelled_choice_row(
                    VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=JOINT_GAIN, sample_index=2
                ),
                labelled_choice_row(
                    VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=OWN_GAIN, sample_index=3
                ),
                labelled_choice_row(
                    VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=None, sample_index=4
                ),
                labelled_choice_row(
                    OTHER_VALUES_ITEM,
                    labels=(JOINT_GAIN, EQUALITY),
                    chosen=JOINT_GAIN,
                    sample_index=0,
                ),
                labelled_choice_row(
                    OTHER_VALUES_ITEM,
                    labels=(JOINT_GAIN, EQUALITY),
                    chosen=EQUALITY,
                    sample_index=1,
                ),
            ],
        )
        by_label = {row["label"]: row for row in values_ordering(battery)}
        assert by_label[JOINT_GAIN] == {
            "step": 0,
            "label": JOINT_GAIN,
            "role": battery_tables.VALUES_POLE_ROLE,
            "rank": "1",
            "win rate (0.5 = chance)": "0.625",
            "items offering": "2",
            "chosen/parsed renders": "4/6",
        }
        assert by_label[EQUALITY]["win rate (0.5 = chance)"] == "0.500"
        assert by_label[EQUALITY]["rank"] == "2"
        assert by_label[OWN_GAIN]["win rate (0.5 = chance)"] == "0.250"
        assert by_label[OWN_GAIN]["rank"] == "3"

    def test_another_familys_labelled_item_never_reaches_the_ordering(self, tmp_path: Path):
        """The whole reason the table filters by family before reading anything.

        The reduction pools by label STRING, and two other families label their options -- the
        graded-dimension menu here, and the risk ladders, whose rungs are option labels too. Unfiltered,
        this cell's ordering carries `own-payoff` beside `joint-gain` and nothing in the rendered table
        says the two answer different questions. Asserted on the denominators as well as the key set,
        because a foreign item sharing a label would move a rate rather than add a row.
        """
        battery = tmp_path / "battery-values-foreign"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                labelled_choice_row(VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=JOINT_GAIN),
                labelled_choice_row(
                    GRADED_CHOICE_ITEM,
                    labels=(GRADED_LABEL, OTHER_GRADED_LABEL, JOINT_GAIN),
                    chosen=JOINT_GAIN,
                    family=FAMILY_GRADED_DIMENSION_AWARENESS,
                ),
            ],
        )
        rows = values_ordering(battery)
        assert sorted(row["label"] for row in rows) == [JOINT_GAIN, OWN_GAIN]
        joint = next(row for row in rows if row["label"] == JOINT_GAIN)
        assert joint["items offering"] == "1"
        assert joint["chosen/parsed renders"] == "1/1"

    def test_the_pooled_pole_is_recomputed_over_its_items_rather_than_averaged(
        self, tmp_path: Path
    ):
        """Three non-social items with different counts, so the two ways of pooling disagree.

        `values-one` offers brevity and is answered for it twice, so brevity's own rate is 1.0.
        `values-two` and `values-three` both offer honesty, one answered for it and one against, so
        honesty's rate is 0.5. Averaging the two per-good rates gives 0.75. The registered reading
        recomputes over the three items -- (1.0 + 1.0 + 0.0) / 3 -- and reads 0.667 over 3 items and 4
        renders. The two per-good rows stay in the table marked as components, because a reader who
        cannot see them cannot tell which good moved.
        """
        battery = tmp_path / "battery-values-pole"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                labelled_choice_row(
                    VALUES_ITEM,
                    labels=(NON_SOCIAL_BREVITY, JOINT_GAIN),
                    chosen=NON_SOCIAL_BREVITY,
                    sample_index=0,
                ),
                labelled_choice_row(
                    VALUES_ITEM,
                    labels=(NON_SOCIAL_BREVITY, JOINT_GAIN),
                    chosen=NON_SOCIAL_BREVITY,
                    sample_index=1,
                ),
                labelled_choice_row(
                    OTHER_VALUES_ITEM,
                    labels=(NON_SOCIAL_HONESTY, EQUALITY),
                    chosen=NON_SOCIAL_HONESTY,
                ),
                labelled_choice_row(
                    THIRD_VALUES_ITEM,
                    labels=(NON_SOCIAL_HONESTY, OWN_GAIN),
                    chosen=OWN_GAIN,
                ),
            ],
        )
        by_label = {row["label"]: row for row in values_ordering(battery)}
        assert by_label[POOLED_NON_SOCIAL]["win rate (0.5 = chance)"] == "0.667"
        assert by_label[POOLED_NON_SOCIAL]["items offering"] == "3"
        assert by_label[POOLED_NON_SOCIAL]["chosen/parsed renders"] == "3/4"
        assert by_label[POOLED_NON_SOCIAL]["role"] == battery_tables.VALUES_POLE_ROLE
        assert by_label[NON_SOCIAL_BREVITY]["win rate (0.5 = chance)"] == "1.000"
        assert by_label[NON_SOCIAL_HONESTY]["win rate (0.5 = chance)"] == "0.500"

    def test_the_components_are_marked_and_left_out_of_the_ranking(self, tmp_path: Path):
        """Ranking the three goods alongside the poles would seat three of five poles in one stretch of
        the ordering and count non-social three times, so the components carry no rank at all."""
        battery = tmp_path / "battery-values-components"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                labelled_choice_row(
                    VALUES_ITEM,
                    labels=(NON_SOCIAL_BREVITY, JOINT_GAIN),
                    chosen=NON_SOCIAL_BREVITY,
                ),
                labelled_choice_row(
                    OTHER_VALUES_ITEM,
                    labels=(NON_SOCIAL_HONESTY, OWN_GAIN),
                    chosen=OWN_GAIN,
                ),
            ],
        )
        rows = values_ordering(battery)
        components = {
            row["label"] for row in rows if row["role"] != battery_tables.VALUES_POLE_ROLE
        }
        assert components == {NON_SOCIAL_BREVITY, NON_SOCIAL_HONESTY}
        assert all(
            row["rank"] == battery_tables.EMPTY_CELL for row in rows if row["label"] in components
        )
        ranked = sorted(
            row["rank"] for row in rows if row["role"] == battery_tables.VALUES_POLE_ROLE
        )
        assert ranked == ["1", "2", "3"]
        assert {row["role"] for row in rows if row["label"] in components} == {
            battery_tables.VALUES_COMPONENT_ROLE
        }

    def test_two_poles_chosen_at_the_same_rate_share_a_rank(self, tmp_path: Path):
        """One item answered once each way. Printing 1 and 2 there would read as an ordering the
        numbers do not contain."""
        battery = tmp_path / "battery-values-tie"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                labelled_choice_row(
                    VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=JOINT_GAIN, sample_index=0
                ),
                labelled_choice_row(
                    VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=OWN_GAIN, sample_index=1
                ),
            ],
        )
        assert [row["rank"] for row in values_ordering(battery)] == ["1", "1"]

    def test_a_cell_that_asked_no_values_item_produces_no_rows(self, tmp_path: Path):
        battery = tmp_path / "battery-values-absent"
        write_cell(
            battery, GROUP_ARM, 0, rows=[control_choice_row(CONTROL_CHOICE_ITEM, canonical_index=0)]
        )
        assert values_ordering(battery) == []

    def test_the_prose_names_the_chance_baseline_and_the_pooling_rule(self, complete_battery: Path):
        document = render(complete_battery)
        assert "value poles, the forced-choice preference ordering" in document
        assert "0.5 is chance on every row" in document
        assert "recomputed over all of their items rather than averaged" in document


class TestTheNominalChoiceSplit:
    def three_family_cell(self, battery: Path) -> None:
        """One cell holding a control item, a values item and a graded-dimension item, all lettered."""
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                control_choice_row(CONTROL_CHOICE_ITEM, canonical_index=0),
                control_choice_row(CONTROL_CHOICE_ITEM, canonical_index=1, sample_index=1),
                labelled_choice_row(VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=JOINT_GAIN),
                labelled_choice_row(
                    GRADED_CHOICE_ITEM,
                    labels=(GRADED_LABEL, OTHER_GRADED_LABEL),
                    chosen=GRADED_LABEL,
                    family=FAMILY_GRADED_DIMENSION_AWARENESS,
                ),
            ],
        )

    def test_the_negative_control_table_refuses_a_target_family_row(self, tmp_path: Path):
        """The mechanism scores every nominal-choice item in the battery, and two target families
        answer in that shape, so without the family filter twenty values items and nine
        graded-dimension items land under a heading whose prose calls them drift placebos. That
        inverts what they measure: a values item moving is the family's registered result, and a
        control moving is a threat to every other reading in the document.
        """
        battery = tmp_path / "battery-control-scope"
        self.three_family_cell(battery)
        controls = battery_tables.build_readout(battery).arms[0].self_report_control_distributions
        assert [row["item"] for row in controls] == [CONTROL_CHOICE_ITEM]
        assert "family" not in controls[0]

    def test_the_target_families_reach_their_own_table_named(self, tmp_path: Path):
        """Complement rather than a named list, so a family landing later in this answer shape appears
        without anyone remembering to add it. The family column is what keeps two readings apart once
        both are in one table."""
        battery = tmp_path / "battery-target-scope"
        self.three_family_cell(battery)
        rows = battery_tables.build_readout(battery).arms[0].self_report_nominal_choices
        assert {row["item"]: row["family"] for row in rows} == {
            GRADED_CHOICE_ITEM: FAMILY_GRADED_DIMENSION_AWARENESS,
            VALUES_ITEM: FAMILY_VALUES_FORCED_CHOICE,
        }

    def test_a_labelled_items_distribution_reads_by_label(self, tmp_path: Path):
        """The values items vary their label order item by item on purpose, so option 0 names a
        different pole on each item of a pole pair and an index distribution would read as a
        preference reversal between the two."""
        battery = tmp_path / "battery-label-keyed"
        self.three_family_cell(battery)
        rows = battery_tables.build_readout(battery).arms[0].self_report_nominal_choices
        values = next(row for row in rows if row["item"] == VALUES_ITEM)
        assert values["distribution"] == f"{JOINT_GAIN}:1.00"

    def test_an_unlabelled_control_keeps_its_option_numbers(self, tmp_path: Path):
        battery = tmp_path / "battery-index-keyed"
        self.three_family_cell(battery)
        controls = battery_tables.build_readout(battery).arms[0].self_report_control_distributions
        assert controls[0]["distribution"] == "0:0.50, 1:0.50"
        assert controls[0]["entropy (bits)"] == "1.00"

    def test_an_item_whose_labels_disagree_inside_one_cell_falls_back_to_numbers(
        self, tmp_path: Path
    ):
        """Labels that moved within one cell cannot name a share, for the same reason a forced-tag item
        whose recorded menu width disagrees with itself reports no width."""
        battery = tmp_path / "battery-label-disagreement"
        write_cell(
            battery,
            GROUP_ARM,
            0,
            rows=[
                labelled_choice_row(VALUES_ITEM, labels=(JOINT_GAIN, OWN_GAIN), chosen=JOINT_GAIN),
                labelled_choice_row(
                    VALUES_ITEM,
                    labels=(EQUALITY, OWN_GAIN),
                    chosen=OWN_GAIN,
                    sample_index=1,
                ),
            ],
        )
        rows = battery_tables.build_readout(battery).arms[0].self_report_nominal_choices
        assert rows[0]["distribution"] == "0:0.50, 1:0.50"

    def test_an_item_answered_one_way_reports_zero_headroom_and_not_a_negative_one(
        self, tmp_path: Path
    ):
        """Same negative-zero bug as the tagged table's, from the same `choice_response_entropy`."""
        battery = tmp_path / "battery-nominal-zero-headroom"
        self.three_family_cell(battery)
        rows = battery_tables.build_readout(battery).arms[0].self_report_nominal_choices
        assert {row["entropy (bits)"] for row in rows} == {"0.00"}

    def test_the_prose_scopes_the_control_heading_and_names_the_other_table(
        self, complete_battery: Path
    ):
        document = render(complete_battery)
        assert "negative controls, per item" in document
        assert "The negative-control family and nothing else" in document
        assert "nominal-choice items outside the controls, per item" in document
        assert "These rows are NOT placebos" in document


class TestContrasts:
    def test_arms_sharing_a_game_are_grouped_by_what_differs(self):
        groups = battery_tables.contrast_groups([GROUP_ARM, SELF_ARM, RUNG_ARM])
        assert [group.game_id for group in groups] == [TWIN_GAME]
        assert groups[0].arms == (GROUP_ARM, SELF_ARM)
        assert groups[0].axis == "grading rule"

    def test_a_lone_arm_on_a_game_is_not_a_contrast(self):
        assert battery_tables.contrast_groups([GROUP_ARM, DICTATOR_ARM]) == []

    def test_the_contrast_section_puts_the_pair_side_by_side(self, complete_battery: Path):
        document = render(complete_battery)
        assert "## Contrasts: arms that trained the same game" in document
        assert f"### `{TWIN_GAME}`: `{GROUP_ARM}`, `{SELF_ARM}`" in document
        assert "Axis that differs: grading rule" in document
        assert f"| step | {GROUP_ARM} | {SELF_ARM} |" in document

    def test_the_transfer_contrast_names_each_arm_own_late_window(self, tmp_path: Path):
        battery = tmp_path / "battery-transfercontrast"
        for step in (0, 10, 20):
            write_cell(battery, GROUP_ARM, step)
        write_cell(battery, SELF_ARM, 0)
        write_cell(battery, SELF_ARM, 10)
        readout = battery_tables.build_readout(battery)
        by_arm = {arm.arm: arm for arm in readout.arms}
        rows = battery_readout._contrast_transfer_rows([by_arm[GROUP_ARM], by_arm[SELF_ARM]])
        assert f"{GROUP_ARM} (late 10,20)" in rows[0]
        assert f"{SELF_ARM} (late 10)" in rows[0]
        assert rows[0]["game"] == TWIN_GAME
        assert rows[0]["role"] == battery_tables.ROLE_TRAINED
        document = battery_readout.render_markdown(readout, command="regenerate-me")
        assert "transfer, base step 0 to each arm's own pooled late window" in document

    def test_byte_identical_sections_are_named_as_shared_draws(self, complete_battery: Path):
        readout = battery_tables.build_readout(complete_battery)
        draws = readout.contrasts[0].shared_draws
        assert {(draw.step, draw.section) for draw in draws} == {
            (step, section)
            for step in (0, 10)
            for section in (SECTION_GAME_BEHAVIOR, SECTION_DT_PROBES, SECTION_CAPABILITIES)
        }
        assert all(draw.arms == (GROUP_ARM, SELF_ARM) for draw in draws)
        document = battery_readout.render_markdown(readout, command="regenerate-me")
        assert "**Shared draws.**" in document
        n_game_rows = sum(1 for row in default_rows() if row["record"] == SECTION_GAME_BEHAVIOR)
        assert (
            f"section `{SECTION_GAME_BEHAVIOR}`: all {n_game_rows} completions byte-identical "
            f"across `{GROUP_ARM}`, `{SELF_ARM}`" in document
        )

    def test_arms_with_their_own_completions_share_no_draw(self, tmp_path: Path):
        battery = tmp_path / "battery-owndraws"
        for arm in (GROUP_ARM, SELF_ARM):
            rows = default_rows()
            for row in rows:
                row["completion"] = f"completion-of-{arm}"
            write_cell(battery, arm, 0, rows=rows)
        readout = battery_tables.build_readout(battery)
        assert readout.contrasts[0].shared_draws == ()
        document = battery_readout.render_markdown(readout, command="regenerate-me")
        assert "Shared draws" not in document

    def test_a_step_one_arm_has_not_reached_renders_empty(self, tmp_path: Path):
        battery = tmp_path / "battery-pairgap"
        write_cell(battery, GROUP_ARM, 0)
        write_cell(battery, GROUP_ARM, 10)
        write_cell(battery, SELF_ARM, 0)
        readout = battery_tables.build_readout(battery)
        by_arm = {arm.arm: arm for arm in readout.arms}
        rows = battery_readout._contrast_table(
            [by_arm[GROUP_ARM], by_arm[SELF_ARM]], battery_readout._contrast_trained
        )
        assert rows[1][SELF_ARM] == battery_tables.EMPTY_CELL


class TestProblemBanner:
    def test_a_disagreeing_git_sha_is_bannered(self, complete_battery: Path):
        write_cell(
            complete_battery, GROUP_ARM, 20, meta=meta_record(GROUP_ARM, 20, git_sha="deadbee")
        )
        write_cell(complete_battery, SELF_ARM, 20)
        document = render(complete_battery)
        assert "PROVENANCE DISAGREEMENT on `git_sha`" in document
        assert "deadbee" in document

    def test_a_section_that_barely_parsed_is_bannered(self, tmp_path: Path):
        battery = tmp_path / "battery-parsefail"
        rows = [game_row(TWIN_GAME, f"twin-{index}", coop=None) for index in range(4)]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        assert "HIGH PARSE FAILURE" in render(battery)

    def test_a_section_running_into_the_cap_is_bannered(self, tmp_path: Path):
        battery = tmp_path / "battery-cap"
        rows = [game_row(TWIN_GAME, "twin-1", coop=None, truncated=True)]
        rows.extend(game_row(TWIN_GAME, f"twin-{index}") for index in range(2, 6))
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        assert "HIGH TRUNCATION" in render(battery)

    def test_a_complete_battery_with_problems_does_not_claim_cells_are_missing(
        self, tmp_path: Path
    ):
        battery = tmp_path / "battery-complete-caveated"
        rows = [game_row(TWIN_GAME, "twin-1", coop=None, truncated=True)]
        rows.extend(game_row(TWIN_GAME, f"twin-{index}") for index in range(2, 6))
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        document = render(battery)
        assert "COMPLETE BUT ERROR-CONTAINING" in document
        assert "remaining cells land" not in document

    def test_an_unregistered_arm_is_bannered_and_has_no_trained_tables(self, tmp_path: Path):
        battery = tmp_path / "battery-unregistered"
        write_cell(battery, UNREGISTERED_ARM, 0)
        readout = battery_tables.build_readout(battery)
        assert readout.arms[0].trained_game is None
        assert readout.arms[0].trained_ladder == []
        assert readout.arms[0].variant_splits == []
        document = battery_readout.render_markdown(readout, command="regenerate-me")
        assert f"UNREGISTERED arm `{UNREGISTERED_ARM}`" in document


class TestDocumentShape:
    def test_the_header_says_it_is_generated_and_how_to_regenerate_it(self, complete_battery: Path):
        document = render(complete_battery)
        assert "GENERATED -- DO NOT HAND-EDIT" in document
        assert "Regenerate with: `regenerate-me`" in document
        assert f"Battery directory: `{complete_battery}`" in document

    def test_the_header_states_the_ground_rules(self, complete_battery: Path):
        document = render(complete_battery)
        assert "0.467" in document
        assert "rate (parsed/asked prompts)" in document
        assert "64% of true" in document
        assert "un-adapted base model" in document
        assert "merge artifact" in document

    def test_every_arm_gets_every_section(self, complete_battery: Path):
        document = render(complete_battery)
        for heading in (
            "trained-game trajectory",
            "transfer matrix, every game evaluated",
            "negative control ladder",
            "decision-theory probes",
            "theory endorsement per multiple-choice render",
            "capability canary",
            "mirrored gamble ladders (loss aversion)",
            "counterpart gaps, an AI counterpart against a human one",
            "cheap talk, announcement against action",
            "trained game split by payoff variant",
            "trained game split by reskin",
            "per-round profile",
            "section health",
        ):
            assert document.count(heading) == 2, heading

    def test_every_computed_table_reaches_the_document(self):
        """A table computed and never printed is the failure this document exists to prevent.

        The readout's own version of the callback that wrote into a dict `Trainer.log` had already
        copied: the number was computed correctly, nothing printed it, and the run had to be repeated.
        A field added to `ArmReadout` without a `markdown_table` call goes red here.
        """
        source = Path(battery_readout.__file__).read_text(encoding="utf-8")
        tables = [
            field.name
            for field in dataclasses.fields(battery_tables.ArmReadout)
            if field.type == "list[dict[str, Any]]"
        ]
        assert len(tables) > 1
        assert [name for name in tables if f"arm.{name}" not in source] == []

    def test_an_empty_table_says_so_rather_than_vanishing(self):
        assert battery_readout.markdown_table([]) == "_(no rows)_"

    def test_a_pipe_in_a_value_cannot_split_a_column(self):
        table = battery_readout.markdown_table([{"game": "a|b"}])
        assert "a\\|b" in table


class TestCli:
    def test_the_cli_writes_the_document_where_it_was_told(
        self, complete_battery: Path, tmp_path: Path
    ):
        out = tmp_path / "nested" / "readout.md"
        battery_readout.main(["--battery-dir", str(complete_battery), "--out", str(out)])
        assert "# Games battery readout" in out.read_text(encoding="utf-8")
        assert f"--out {out}" in out.read_text(encoding="utf-8")

    def test_the_default_output_name_comes_from_the_battery_directory(self):
        assert battery_readout.default_out_path(Path("artifacts/x/battery-3e7f227")) == (
            battery_readout.DEFAULT_OUT_DIR / "games-readout-3e7f227.md"
        )

    def test_an_out_path_on_trackable_repo_ground_is_refused(
        self, complete_battery: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        monkeypatch.chdir(repo_root)
        with pytest.raises(ValueError, match="per-item probe results"):
            battery_readout.write_readout(complete_battery, repo_root / "games" / "readout.md")

    def test_out_paths_in_scratch_artifacts_or_outside_the_repo_are_allowed(
        self, complete_battery: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        monkeypatch.chdir(repo_root)
        for out in (
            repo_root / "docs" / "scratch" / "readout.md",
            repo_root / "artifacts" / "readout.md",
            tmp_path / "elsewhere" / "readout.md",
        ):
            assert battery_readout.write_readout(complete_battery, out) == out

    def test_the_default_battery_is_the_most_recently_written_one(self, tmp_path: Path):
        older = tmp_path / "battery-older"
        newer = tmp_path / "battery-newer"
        for path, mtime in ((older, 1_000_000), (newer, 2_000_000)):
            write_cell(path, GROUP_ARM, 0)
            os.utime(path, (mtime, mtime))
        assert battery_cells.latest_battery_dir(tmp_path) == newer

    def test_a_root_with_no_battery_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no battery to read"):
            battery_cells.latest_battery_dir(tmp_path)
