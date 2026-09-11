"""Pin the report tables against synthetic traces whose answers are known exactly.

Offline and CPU-only: every trace here is written by hand, so each table's expected numbers are
arithmetic rather than a regenerated fixture.

:class:`TestAnAllFdtTraceReportsAsAllFdt` is the plan's Phase 2 sabotage item at the reporting
end. A pipeline that averaged wrongly, dropped unparsed rows into a default, or mislabelled a
checkpoint would still emit a well-formed table, so the check is to feed it a trace whose
distribution is 100% one theory and require exactly that back.

:class:`TestCheckpointOrderingAndLabels` covers the two ways a comparison table goes silently
wrong: rows ordered 10, 100, 20 read as a non-monotonic trend that is purely a sorting artefact,
and a row labelled from a filename can attribute a number to the wrong arm while looking right.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from games.arms import ARMS, arm_game_ids
from games.evals import (
    RECORD_META,
    SECTION_CAPABILITIES,
    SECTION_DT_PROBES,
    SECTION_GAME_BEHAVIOR,
)
from games.report import (
    action_rate_matrix,
    capability_table,
    load_traces,
    per_round_profile,
    render_report,
    sample_cot_excerpts,
    theory_item_flips,
    theory_shift_table,
)


def write_trace(  # noqa: PLR0913 - one keyword per meta field a test here needs to vary
    path: Path,
    *,
    arm: str,
    step: int,
    records: list[dict[str, Any]],
    backend_kind: str = "hf",
    eval_config: dict[str, Any] | None = None,
) -> Path:
    """Write one synthetic eval trace, meta record first."""
    meta: dict[str, Any] = {
        "record": RECORD_META,
        "written_at": "2026-08-17T00:00:00+00:00",
        "git_sha": "synthetic",
        "backend_model_id": "synthetic",
        "backend_kind": backend_kind,
        "sections": [SECTION_GAME_BEHAVIOR, SECTION_DT_PROBES, SECTION_CAPABILITIES],
        "eval_config": eval_config if eval_config is not None else {},
        "arm": arm,
        "step": step,
    }
    return write_records(path, [meta, *records])


def write_records(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write records verbatim, which is how the missing-meta cases are built."""
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


_DISTINCT_PROMPT_INDICES = itertools.count()


def game_record(game_id: str, **overrides: Any) -> dict[str, Any]:
    """One game-behaviour record with just the fields the report reads.

    Every call is a DIFFERENT prompt by default, because the report averages within a prompt before
    averaging over prompts: a helper handing out one shared id would turn three records meant as
    three prompts into three draws of one, and the tables would silently read one observation where
    the test wrote three. Pass `prompt_id` explicitly to express several draws of one prompt, which
    is what the battery does at `game_behavior_samples > 1`.
    """
    record: dict[str, Any] = {
        "record": SECTION_GAME_BEHAVIOR,
        "game_id": game_id,
        "prompt_id": f"{game_id}--frame{next(_DISTINCT_PROMPT_INDICES)}--coop0",
        "sample_index": 0,
        "trained_game": True,
        "parsed": True,
        "coop_fraction": None,
        "moves": None,
        "completion": "",
        "truncated_thinking": False,
    }
    record.update(overrides)
    return record


def probe_record(probe_id: str, **overrides: Any) -> dict[str, Any]:
    """One decision-theory probe record, open-ended when a `theory` override is given."""
    record: dict[str, Any] = {
        "record": SECTION_DT_PROBES,
        "probe_id": probe_id,
        "source": "dtbench" if probe_id.startswith("dtbench-") else "ours",
        "family": "test",
        "kind": "open-ended" if overrides.get("theory") is not None else "multiple-choice",
        "sample_index": 0,
        "theory": None,
        "compatible_theories": None,
        "edt_leaning": None,
        "parsed": True,
        "completion": "",
        "truncated_thinking": False,
    }
    record.update(overrides)
    return record


def capability_record(*, correct: bool) -> dict[str, Any]:
    """One arithmetic-canary record."""
    return {
        "record": SECTION_CAPABILITIES,
        "item_index": 0,
        "correct": correct,
        "parsed": True,
        "completion": "",
        "truncated_thinking": False,
    }


class TestAnAllFdtTraceReportsAsAllFdt:
    def test_the_distribution_is_exactly_one_theory(self, tmp_path: Path) -> None:
        """The multiple-choice records are load-bearing: they carry no theory of their own.

        A fraction taken over every probe record rather than over the answered open-ended ones
        would report 0.75 here and look like a plausible distribution.
        """
        path = write_trace(
            tmp_path / "fdt.jsonl",
            arm="twin-pd-self",
            step=40,
            records=[
                *(probe_record(f"probe-{index}", theory="FDT") for index in range(12)),
                *(
                    probe_record(
                        f"dtbench-{index}.1ATT", compatible_theories=["CDT"], edt_leaning=-1
                    )
                    for index in range(4)
                ),
            ],
        )
        table = theory_shift_table(load_traces([path]))
        assert len(table) == 1
        assert table.loc[0, "frac_FDT"] == pytest.approx(1.0)
        assert table.loc[0, "n_open_ended"] == 12
        assert table.loc[0, "n_scored_items"] == 4
        assert [column for column in table.columns if column.startswith("frac_")] == ["frac_FDT"]

    def test_a_mixed_trace_reports_the_mix(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "mixed.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                probe_record("a", theory="CDT"),
                probe_record("b", theory="CDT"),
                probe_record("c", theory="FDT"),
                probe_record("d", theory="EDT"),
            ],
        )
        table = theory_shift_table(load_traces([path]))
        assert table.loc[0, "frac_CDT"] == pytest.approx(0.5)
        assert table.loc[0, "frac_FDT"] == pytest.approx(0.25)
        assert table.loc[0, "frac_EDT"] == pytest.approx(0.25)

    def test_the_edt_leaning_scalar_comes_from_the_choice_items(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "leaning.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                probe_record("a", compatible_theories=["EDT"], edt_leaning=1),
                probe_record("b", compatible_theories=["CDT"], edt_leaning=-1),
                probe_record("c", compatible_theories=["CDT", "EDT"], edt_leaning=0),
            ],
        )
        table = theory_shift_table(load_traces([path]))
        assert table.loc[0, "mean_edt_leaning"] == pytest.approx(0.0)
        assert table.loc[0, "n_scored_items"] == 3


class TestCounterbalancedItemsCountOnce:
    """Every choice item is now asked under both option orders, so a render is not an observation.

    `games.evals` reduces to one value per item before any mean. The report pooled renders instead,
    which doubled the denominator and tilted the mean toward whichever items happened to be
    order-stable -- and in the flip table it was worse than a lost datum: keying the per-item answer
    on probe_id alone kept whichever order was written last, so an order-sensitive item could
    register a between-checkpoint "flip" that was really a coin landing differently in two traces.
    """

    def _counterbalanced(self, probe_id: str, *, leanings: tuple[float, float]) -> list[Any]:
        """One item under both orders, with the two renders' leanings given in order.

        `answer_index` is the CANONICAL option index, so an item answered the same way under both
        orders carries the same index twice, which is what order disagreement keys on.
        """
        return [
            probe_record(
                probe_id,
                option_order_name=name,
                answer_index=0 if leaning > 0 else 1,
                compatible_theories=["EDT"] if leaning > 0 else ["CDT"],
                edt_leaning=leaning,
            )
            for name, leaning in (("as-authored", leanings[0]), ("reversed", leanings[1]))
        ]

    def test_the_denominator_counts_items_not_renders(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "counterbalanced.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                *self._counterbalanced("dtbench-1.1ATT", leanings=(1.0, 1.0)),
                *self._counterbalanced("dtbench-1.2ATT", leanings=(-1.0, -1.0)),
            ],
        )
        table = theory_shift_table(load_traces([path]))
        assert table.loc[0, "n_scored_items"] == 2
        assert table.loc[0, "mean_edt_leaning"] == pytest.approx(0.0)

    def test_an_order_flipped_item_weighs_the_same_as_a_stable_one(self, tmp_path: Path) -> None:
        """Pooling renders let a stable item count twice against a flipped item's two halves."""
        path = write_trace(
            tmp_path / "flipped.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                *self._counterbalanced("dtbench-1.1ATT", leanings=(1.0, 1.0)),
                *self._counterbalanced("dtbench-1.2ATT", leanings=(1.0, -1.0)),
            ],
        )
        table = theory_shift_table(load_traces([path]))
        assert table.loc[0, "n_scored_items"] == 2
        assert table.loc[0, "mean_edt_leaning"] == pytest.approx(0.5)

    def test_the_order_disagreement_rate_travels_with_the_mean(self, tmp_path: Path) -> None:
        """A model answering by letter position scores 1.0 here, which is what stops it reading
        as a decision-theory position."""
        path = write_trace(
            tmp_path / "disagreeing.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                *self._counterbalanced("dtbench-1.1ATT", leanings=(1.0, 1.0)),
                *self._counterbalanced("dtbench-1.2ATT", leanings=(1.0, -1.0)),
            ],
        )
        table = theory_shift_table(load_traces([path]))
        assert table.loc[0, "order_disagreement_rate"] == pytest.approx(0.5)

    def test_an_item_that_answers_differently_under_two_orders_cannot_flip(
        self, tmp_path: Path
    ) -> None:
        """It has not told us what it endorses, so it is unscored rather than a datum.

        Before this, `answers_by_probe` kept whichever render was written last, so the same
        order-sensitive item read as EDT in one trace and CDT in the next and the flip table
        reported a between-checkpoint change that no checkpoint made.
        """
        before = write_trace(
            tmp_path / "before.jsonl",
            arm="twin-pd-group",
            step=0,
            records=[
                probe_record(
                    "dtbench-1.1ATT",
                    option_order_name="as-authored",
                    answer_index=0,
                    compatible_theories=["EDT"],
                    edt_leaning=1,
                ),
                probe_record(
                    "dtbench-1.1ATT",
                    option_order_name="reversed",
                    answer_index=1,
                    compatible_theories=["CDT"],
                    edt_leaning=-1,
                ),
            ],
        )
        after = write_trace(
            tmp_path / "after.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                probe_record(
                    "dtbench-1.1ATT",
                    option_order_name="as-authored",
                    answer_index=1,
                    compatible_theories=["CDT"],
                    edt_leaning=-1,
                ),
                probe_record(
                    "dtbench-1.1ATT",
                    option_order_name="reversed",
                    answer_index=0,
                    compatible_theories=["EDT"],
                    edt_leaning=1,
                ),
            ],
        )
        assert theory_item_flips(load_traces([before, after])).empty

    def test_an_item_stable_under_both_orders_still_flips_when_it_changes(
        self, tmp_path: Path
    ) -> None:
        """The reduction must not cost the readout its actual signal."""
        before = write_trace(
            tmp_path / "stable-before.jsonl",
            arm="twin-pd-group",
            step=0,
            records=self._counterbalanced("dtbench-1.1ATT", leanings=(1.0, 1.0)),
        )
        after = write_trace(
            tmp_path / "stable-after.jsonl",
            arm="twin-pd-group",
            step=40,
            records=self._counterbalanced("dtbench-1.1ATT", leanings=(-1.0, -1.0)),
        )
        flips = theory_item_flips(load_traces([before, after]))
        assert list(flips["probe_id"]) == ["dtbench-1.1ATT"]
        assert flips.loc[0, "before"] == "EDT"
        assert flips.loc[0, "after"] == "CDT"


class TestCheckpointOrderingAndLabels:
    def test_traces_come_back_in_numeric_step_order(self, tmp_path: Path) -> None:
        paths = [
            write_trace(
                tmp_path / f"s{step}.jsonl",
                arm="twin-pd-group",
                step=step,
                records=[probe_record("a", theory="CDT")],
            )
            for step in (100, 10, 20)
        ]
        assert [trace.step for trace in load_traces(paths)] == [10, 20, 100]

    def test_arms_are_grouped_before_steps(self, tmp_path: Path) -> None:
        paths = [
            write_trace(
                tmp_path / "b20.jsonl",
                arm="dictator",
                step=20,
                records=[capability_record(correct=True)],
            ),
            write_trace(
                tmp_path / "a10.jsonl",
                arm="twin-pd-group",
                step=10,
                records=[capability_record(correct=True)],
            ),
            write_trace(
                tmp_path / "b10.jsonl",
                arm="dictator",
                step=10,
                records=[capability_record(correct=True)],
            ),
        ]
        assert [(trace.arm, trace.step) for trace in load_traces(paths)] == [
            ("dictator", 10),
            ("dictator", 20),
            ("twin-pd-group", 10),
        ]

    def test_a_trace_without_a_meta_record_raises(self, tmp_path: Path) -> None:
        path = write_records(tmp_path / "headless.jsonl", [probe_record("a", theory="CDT")])
        with pytest.raises(ValueError, match="does not start with"):
            load_traces([path])

    @pytest.mark.parametrize("missing", ["arm", "step"])
    def test_a_meta_record_missing_its_labels_raises(self, tmp_path: Path, missing: str) -> None:
        """Labelling a row from the filename instead would be silently wrong, not obviously so."""
        path = tmp_path / "unlabelled.jsonl"
        meta = {
            "record": RECORD_META,
            "git_sha": "synthetic",
            "arm": "twin-pd-group",
            "step": 10,
        }
        del meta[missing]
        path.write_text(json.dumps(meta) + "\n")
        with pytest.raises(ValueError, match="missing"):
            load_traces([path])

    def test_an_empty_path_list_raises(self) -> None:
        with pytest.raises(ValueError, match="eval_paths is empty"):
            load_traces([])


class TestActionRateMatrix:
    def test_rates_and_counts_are_per_game(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "games.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                game_record("twin-pd", coop_fraction=1.0),
                game_record("twin-pd", coop_fraction=0.0),
                game_record("twin-pd", coop_fraction=0.0),
                game_record("public-goods", coop_fraction=1.0, trained_game=False),
            ],
        )
        table = action_rate_matrix(load_traces([path])).set_index("game_id")
        assert table.loc["twin-pd", "coop_rate"] == pytest.approx(1 / 3)
        assert table.loc["twin-pd", "n_parsed"] == 3
        assert table.loc["public-goods", "coop_rate"] == pytest.approx(1.0)
        assert bool(table.loc["public-goods", "trained_by_this_arm"]) is False

    def test_the_rate_is_a_mean_over_prompts_not_over_draws(self, tmp_path: Path) -> None:
        """Several draws of one prompt are one observation; pooling them lets one prompt outvote another.

        Discriminating the two needs prompts with UNEQUAL parsed draw counts, because with every
        prompt contributing the same number the pooled mean and the mean of per-prompt means are
        arithmetically identical -- a test built on a uniform cell passes under either
        implementation, which is no test at all. Here one prompt answers eight times and its
        neighbour once, so the two averages separate: per prompt it is the mean of 1.0 and 0.0, and
        pooled it is eight ones against one zero.
        """
        path = write_trace(
            tmp_path / "unequal-draws.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                *(
                    game_record(
                        "twin-pd",
                        prompt_id="twin-pd--loud--coop0",
                        sample_index=draw,
                        coop_fraction=1.0,
                    )
                    for draw in range(8)
                ),
                game_record("twin-pd", prompt_id="twin-pd--quiet--coop0", coop_fraction=0.0),
            ],
        )
        table = action_rate_matrix(load_traces([path])).set_index("game_id")
        assert table.loc["twin-pd", "coop_rate"] == pytest.approx(0.5)
        assert table.loc["twin-pd", "coop_rate"] != pytest.approx(8 / 9)
        assert table.loc["twin-pd", "n_parsed"] == 2
        assert table.loc["twin-pd", "n_prompts"] == 2
        assert table.loc["twin-pd", "n_records"] == 9

    def test_a_round_profile_weights_prompts_equally_across_their_draws(
        self, tmp_path: Path
    ) -> None:
        """Same argument one level down: a prompt with more surviving draws must not set the shape.

        The final-round rate is the tell this table exists for, so a prompt sampled eight times
        defecting at the end and a prompt sampled once cooperating there have to weigh the same.
        Pooling the moves would read the last round as 1/9 cooperative instead of 1/2.
        """
        path = write_trace(
            tmp_path / "unequal-rounds.jsonl",
            arm="iterated-pd-tft",
            step=40,
            records=[
                *(
                    game_record(
                        "iterated-pd-tft",
                        prompt_id="iterated--loud--coop0",
                        sample_index=draw,
                        coop_fraction=0.5,
                        moves=["C", "D"],
                    )
                    for draw in range(8)
                ),
                game_record(
                    "iterated-pd-tft",
                    prompt_id="iterated--quiet--coop0",
                    coop_fraction=1.0,
                    moves=["C", "C"],
                ),
            ],
        )
        table = per_round_profile(load_traces([path])).set_index("round_number")
        assert table.loc[2, "coop_rate"] == pytest.approx(0.5)
        assert table.loc[2, "coop_rate"] != pytest.approx(1 / 9)
        assert table.loc[2, "n_prompts"] == 2
        assert table.loc[2, "n_moves"] == 9

    def test_unparsed_completions_lower_the_parse_rate_without_moving_the_coop_rate(
        self, tmp_path: Path
    ) -> None:
        path = write_trace(
            tmp_path / "partial.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                game_record("twin-pd", coop_fraction=1.0),
                game_record("twin-pd", coop_fraction=None, parsed=False),
            ],
        )
        table = action_rate_matrix(load_traces([path])).set_index("game_id")
        assert table.loc["twin-pd", "coop_rate"] == pytest.approx(1.0)
        assert table.loc["twin-pd", "n_parsed"] == 1
        assert table.loc["twin-pd", "n_prompts"] == 2
        assert table.loc["twin-pd", "parse_failure_rate"] == pytest.approx(0.5)

    def test_a_game_nothing_parsed_for_reports_no_rate_rather_than_zero(
        self, tmp_path: Path
    ) -> None:
        """A missing measurement and a measured zero are different claims."""
        path = write_trace(
            tmp_path / "none.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[game_record("twin-pd", coop_fraction=None, parsed=False)],
        )
        table = action_rate_matrix(load_traces([path])).set_index("game_id")
        assert (
            table.loc["twin-pd", "coop_rate"] is None
            or pytest.approx(table.loc["twin-pd", "n_parsed"]) == 0
        )


class TestTheTransferColumnIsPerArmNotPerSlate:
    """The cross-game grid's whole point is which games this arm never trained on.

    `games.evals` sets `trained_game=True` for every game with a training arm *anywhere* in the
    slate, which is right at slate level and wrong in a per-arm table: a twin-pd-group report
    labelled hi-lo, chicken, stag and the rest as trained, burying most of the transfer games in
    the trained bucket and inverting the column's stated purpose. So the sabotage here is a record
    that claims `trained_game=True` for a game the arm never touched, and the table has to
    contradict it.
    """

    def test_the_arm_s_own_game_is_the_only_trained_one(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "slate.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                game_record("twin-pd", coop_fraction=1.0),
                game_record("hi-lo", coop_fraction=0.5),
                game_record("public-goods", coop_fraction=0.0, trained_game=False),
            ],
        )
        table = action_rate_matrix(load_traces([path])).set_index("game_id")
        assert bool(table.loc["twin-pd", "trained_by_this_arm"]) is True
        assert bool(table.loc["hi-lo", "trained_by_this_arm"]) is False
        assert bool(table.loc["public-goods", "trained_by_this_arm"]) is False

    def test_the_record_s_slate_level_flag_does_not_decide(self, tmp_path: Path) -> None:
        """hi-lo has its own arm, so `games.evals` marks it trained in every trace it appears in."""
        path = write_trace(
            tmp_path / "flagged.jsonl",
            arm="twin-pd-self",
            step=40,
            records=[game_record("hi-lo", coop_fraction=0.5, trained_game=True)],
        )
        table = action_rate_matrix(load_traces([path])).set_index("game_id")
        assert bool(table.loc["hi-lo", "trained_by_this_arm"]) is False

    def test_a_breadth_arm_marks_every_game_its_corpus_carried(self, tmp_path: Path) -> None:
        """The same error pointed the other way: an arm whose corpus was six games trains six.

        Reading the lead game alone would file the other five as transfer, and the headline table
        would then read as generalisation to games the run trained on directly.
        """
        path = write_trace(
            tmp_path / "breadth.jsonl",
            arm="prosocial-breadth-care1",
            step=200,
            records=[
                game_record("twin-pd", coop_fraction=1.0),
                game_record("stag-hunt", coop_fraction=0.8),
                game_record("chicken", coop_fraction=0.7),
                game_record("hi-lo", coop_fraction=0.5),
            ],
        )
        table = action_rate_matrix(load_traces([path])).set_index("game_id")
        assert bool(table.loc["twin-pd", "trained_by_this_arm"]) is True
        assert bool(table.loc["stag-hunt", "trained_by_this_arm"]) is True
        assert bool(table.loc["chicken", "trained_by_this_arm"]) is True
        assert bool(table.loc["hi-lo", "trained_by_this_arm"]) is False

    def test_a_label_that_is_not_a_registry_arm_reports_unknown_rather_than_guessing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`--arm` is free text, so a hosted-model eval can carry a label no registry entry has."""
        path = write_trace(
            tmp_path / "adhoc.jsonl",
            arm="some-hosted-model-probe",
            step=0,
            records=[game_record("twin-pd", coop_fraction=1.0)],
        )
        with caplog.at_level(logging.WARNING):
            table = action_rate_matrix(load_traces([path])).set_index("game_id")
        assert table.loc["twin-pd", "trained_by_this_arm"] is None
        assert "some-hosted-model-probe" in caplog.text


class TestACorpusShortOfItsArmsRegistryEntryIsNamedRatherThanBuried:
    """A breadth run can train fewer games than its arm allows, and the transfer column cannot tell.

    The column stays derived from `games.arms.ARMS`, which is a deliberate decision recorded in
    `games.battery_tables`: a step-0 cell correctly declares no trained game, so an arm whose only
    landed cell is step 0 would read as having trained nothing. But `games.breadth_corpus` can drop a
    whole game group at selection, and the cell's own `eval_config.trained_game_ids` (which
    `games.run_evals` reads off the run's corpus composition) then names the shorter truth while the
    registry still allows the dropped game. The table files that game in the trained bucket, so the
    document has to say out loud that the run held no rows of it -- it is the cleanest transfer
    evidence the run produced, and reading it as in-distribution inverts the finding.
    """

    ARM = "prosocial-breadth-care1"
    DROPPED = "chicken"

    @classmethod
    def held_games(cls) -> tuple[str, ...]:
        """Every game the arm allows except the one this fixture's selection dropped whole."""
        return tuple(game for game in arm_game_ids(ARMS[cls.ARM]) if game != cls.DROPPED)

    def test_the_document_names_the_game_the_corpus_never_held(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = write_trace(
            tmp_path / "short-corpus.jsonl",
            arm=self.ARM,
            step=200,
            eval_config={"trained_game_ids": list(self.held_games())},
            records=[game_record(self.DROPPED, coop_fraction=0.7)],
        )
        with caplog.at_level(logging.WARNING):
            markdown = render_report([path])
        note = next(line for line in markdown.splitlines() if "held no rows" in line)
        assert self.DROPPED in note
        assert f"{self.ARM}@200" in note
        assert self.DROPPED in caplog.text

    def test_a_corpus_covering_every_allowed_game_gets_no_note(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "full-corpus.jsonl",
            arm=self.ARM,
            step=200,
            eval_config={"trained_game_ids": list(arm_game_ids(ARMS[self.ARM]))},
            records=[game_record(self.DROPPED, coop_fraction=0.7)],
        )
        assert "held no rows" not in render_report([path])

    def test_a_step_zero_cell_declaring_nothing_is_not_a_short_corpus(self, tmp_path: Path) -> None:
        """The base model trained nothing at all, which is the empty declaration's honest meaning."""
        path = write_trace(
            tmp_path / "base.jsonl",
            arm=self.ARM,
            step=0,
            eval_config={"trained_game_ids": []},
            records=[game_record(self.DROPPED, coop_fraction=0.7, trained_game=False)],
        )
        assert "held no rows" not in render_report([path])


class TestAMockTraceCannotPassForAMeasurement:
    """`--backend mock` samples nothing, and its canned completion is deliberately unparseable.

    So a mock row lands in the table with no coop rate and a parse-failure rate of 1.000, which
    reads exactly like a real termination failure. Nothing in the tables read the meta's
    `backend_kind`, so the report a human opens said nothing about it.
    """

    def test_the_inventory_line_marks_a_mock_trace(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "mock.jsonl",
            arm="twin-pd-group",
            step=5,
            backend_kind="mock",
            records=[game_record("twin-pd", coop_fraction=None, parsed=False)],
        )
        markdown = render_report([path])
        assert "twin-pd-group@5 (mock)" in markdown
        assert "--backend mock" in markdown
        assert "not measurements" in markdown

    def test_a_real_trace_carries_no_such_mark(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "real.jsonl",
            arm="twin-pd-group",
            step=5,
            backend_kind="hf",
            records=[game_record("twin-pd", coop_fraction=1.0)],
        )
        markdown = render_report([path])
        assert "twin-pd-group@5" in markdown
        assert "(mock)" not in markdown
        assert "--backend mock" not in markdown


class TestPerRoundProfile:
    def test_end_game_defection_shows_up_in_the_last_round(self, tmp_path: Path) -> None:
        """The headline plot: cooperation holds through the early rounds and drops at the end."""
        path = write_trace(
            tmp_path / "iterated.jsonl",
            arm="iterated-pd-tft",
            step=40,
            records=[
                game_record("iterated-pd-tft", coop_fraction=0.8, moves=["C", "C", "C", "C", "D"]),
                game_record("iterated-pd-tft", coop_fraction=0.8, moves=["C", "C", "C", "C", "D"]),
                game_record("iterated-pd-tft", coop_fraction=1.0, moves=["C", "C", "C", "C", "C"]),
            ],
        )
        table = per_round_profile(load_traces([path])).set_index("round_number")
        assert table.loc[1, "coop_rate"] == pytest.approx(1.0)
        assert table.loc[4, "coop_rate"] == pytest.approx(1.0)
        assert table.loc[5, "coop_rate"] == pytest.approx(1 / 3)
        assert table.loc[5, "n_moves"] == 3

    def test_records_without_moves_are_skipped(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "mixed.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                game_record("twin-pd", coop_fraction=1.0),
                game_record("iterated-pd-tft", coop_fraction=1.0, moves=["C", "C"]),
            ],
        )
        table = per_round_profile(load_traces([path]))
        assert list(table["round_number"]) == [1, 2]


class TestTheoryItemFlips:
    def test_items_that_changed_answer_are_listed_with_before_and_after(
        self, tmp_path: Path
    ) -> None:
        before = write_trace(
            tmp_path / "before.jsonl",
            arm="twin-pd-group",
            step=0,
            records=[
                probe_record("dtbench-1.1ATT", compatible_theories=["EDT"], edt_leaning=1),
                probe_record("dtbench-1.2ATT", compatible_theories=["CDT"], edt_leaning=-1),
            ],
        )
        after = write_trace(
            tmp_path / "after.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                probe_record("dtbench-1.1ATT", compatible_theories=["CDT"], edt_leaning=-1),
                probe_record("dtbench-1.2ATT", compatible_theories=["CDT"], edt_leaning=-1),
            ],
        )
        flips = theory_item_flips(load_traces([before, after]))
        assert list(flips["probe_id"]) == ["dtbench-1.1ATT"]
        assert flips.loc[0, "before"] == "EDT"
        assert flips.loc[0, "after"] == "CDT"
        assert flips.loc[0, "from_step"] == 0
        assert flips.loc[0, "to_step"] == 40

    def test_an_unchanged_battery_produces_an_empty_table(self, tmp_path: Path) -> None:
        records = [probe_record("a", compatible_theories=["EDT"], edt_leaning=1)]
        before = write_trace(tmp_path / "b.jsonl", arm="arm", step=0, records=records)
        after = write_trace(tmp_path / "a.jsonl", arm="arm", step=40, records=records)
        assert theory_item_flips(load_traces([before, after])).empty

    def test_a_single_checkpoint_has_nothing_to_compare(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "one.jsonl",
            arm="arm",
            step=0,
            records=[probe_record("a", theory="CDT")],
        )
        assert theory_item_flips(load_traces([path])).empty

    def test_open_ended_answers_flip_on_the_matched_theory(self, tmp_path: Path) -> None:
        before = write_trace(
            tmp_path / "b.jsonl", arm="arm", step=0, records=[probe_record("open-a", theory="FDT")]
        )
        after = write_trace(
            tmp_path / "a.jsonl", arm="arm", step=40, records=[probe_record("open-a", theory="CDT")]
        )
        flips = theory_item_flips(load_traces([before, after]))
        assert flips.loc[0, "before"] == "FDT"
        assert flips.loc[0, "after"] == "CDT"


class TestCapabilityTable:
    def test_accuracy_is_the_fraction_correct(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "cap.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                capability_record(correct=True),
                capability_record(correct=True),
                capability_record(correct=False),
                capability_record(correct=False),
            ],
        )
        table = capability_table(load_traces([path]))
        assert table.loc[0, "accuracy"] == pytest.approx(0.5)
        assert table.loc[0, "n_items"] == 4

    def test_a_trace_without_the_section_is_omitted(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "nocap.jsonl",
            arm="arm",
            step=0,
            records=[probe_record("a", theory="CDT")],
        )
        assert capability_table(load_traces([path])).empty


class TestSampleCotExcerpts:
    def test_it_finds_the_reasoning_families_it_is_looking_for(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "cot.jsonl",
            arm="twin-pd-self",
            step=40,
            records=[
                game_record(
                    "twin-pd",
                    coop_fraction=1.0,
                    completion="The other instance will make the same decision as me, so I hold.",
                ),
                game_record(
                    "twin-pd",
                    coop_fraction=0.0,
                    completion="Their choice is already fixed and I cannot affect it now.",
                ),
            ],
        )
        excerpts = sample_cot_excerpts(load_traces([path]))
        families = set(excerpts["family"])
        assert "correlated-reasoning" in families
        assert "causal-separation" in families
        assert all(len(text) > 0 for text in excerpts["excerpt"])

    def test_it_caps_the_number_per_family(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "many.jsonl",
            arm="arm",
            step=0,
            records=[
                game_record("twin-pd", coop_fraction=1.0, completion="the same decision as me")
                for _ in range(10)
            ],
        )
        excerpts = sample_cot_excerpts(load_traces([path]), per_family=2)
        assert len(excerpts[excerpts["family"] == "correlated-reasoning"]) == 2

    def test_custom_families_override_the_defaults(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "custom.jsonl",
            arm="arm",
            step=0,
            records=[game_record("twin-pd", coop_fraction=1.0, completion="a pangolin appeared")],
        )
        excerpts = sample_cot_excerpts(load_traces([path]), families={"zoology": ("pangolin",)})
        assert list(excerpts["family"]) == ["zoology"]
        assert "pangolin" in excerpts.loc[0, "excerpt"]

    def test_a_non_positive_cap_raises(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "c.jsonl", arm="arm", step=0, records=[probe_record("a", theory="CDT")]
        )
        with pytest.raises(ValueError, match="per_family must be at least 1"):
            sample_cot_excerpts(load_traces([path]), per_family=0)


class TestRenderReport:
    def test_it_renders_every_section_as_markdown(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "full.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                game_record("twin-pd", coop_fraction=1.0),
                game_record("public-goods", coop_fraction=0.0, trained_game=False),
                game_record("iterated-pd-tft", coop_fraction=0.8, moves=["C", "C", "C", "C", "D"]),
                probe_record("open-a", theory="FDT", completion="we both reason the same way"),
                probe_record("dtbench-1.1ATT", compatible_theories=["EDT"], edt_leaning=1),
                capability_record(correct=True),
            ],
        )
        markdown = render_report([path])
        for heading in (
            "# Game-theory RL vibe report",
            "## Action rates by game",
            "## Per-round cooperation",
            "## Decision-theory distribution",
            "## Per-item answer flips",
            "## Capability canary",
            "## Reasoning excerpts",
        ):
            assert heading in markdown
        assert "twin-pd-group@40" in markdown
        assert "| arm | step |" in markdown

    def test_empty_tables_render_as_a_note_rather_than_breaking(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "sparse.jsonl",
            arm="arm",
            step=0,
            records=[probe_record("a", theory="CDT")],
        )
        markdown = render_report([path])
        assert "_(no rows)_" in markdown

    def test_a_custom_title_is_used(self, tmp_path: Path) -> None:
        path = write_trace(
            tmp_path / "t.jsonl", arm="arm", step=0, records=[probe_record("a", theory="CDT")]
        )
        assert "# Stag hunt, step 40" in render_report([path], title="Stag hunt, step 40")

    def test_each_trace_is_read_once_rather_than_once_per_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Traces carry every completion's full text and a ladder is fourteen checkpoints.

        Each table function used to re-read the paths it was handed, so one report opened and
        JSON-parsed every trace seven times over -- the largest reader in the eval path doing seven
        times the work with seven transient copies of the records at the peak.
        """
        path = write_trace(
            tmp_path / "read-once.jsonl",
            arm="twin-pd-group",
            step=40,
            records=[
                game_record("twin-pd", coop_fraction=1.0),
                probe_record("open-a", theory="FDT"),
                capability_record(correct=True),
            ],
        )
        opens: list[Path] = []
        original = Path.open

        def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
            opens.append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", counting_open)
        render_report([path])
        assert opens.count(path) == 1
