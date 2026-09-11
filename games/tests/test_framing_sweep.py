"""The counterpart-framing sweep: swapped "About the other side" paragraphs, everything else pinned.

The sweep decomposes the twin clause that the 9B ladder measured as gating cross-game transfer:
which part of "another instance of this same model ... deciding on it the same way you are"
carries the trained disposition. These tests pin the properties the measurement depends on: the
`twin` framing reproduces the trained rendering byte for byte (the within-run reference cell), the
`same-weights-uncorrelated` framing keeps twin's shared-weights assertion word for word so that the
only thing separating those two cells is the denied decision route, the unstated framing really
omits the paragraph, the disclosed-decision framing names the label the grading column calls
cooperative under each row's own mapping, framing rows refuse the training split (so "never trained
under this framing" stays true by construction), and the battery section stamps every record with
the framing that produced it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games.evals import (
    FRAMING_SWEEP_GAME_IDS,
    RECORD_META,
    SECTION_FRAMING_SWEEP,
    EvalConfig,
    plan_battery,
    run_eval_battery,
)
from games.framing_stimulus import (
    RuntimeFramings,
    framing_states_coupling,
    load_framings,
    resolve_framing_clause,
)
from games.prompts import (
    COUNTERPART_FRAMING_IDS,
    COUNTERPART_FRAMINGS,
    DIFFERENT_AI_COUNTERPART_CLAUSE,
    FRAMING_DIFFERENT_AI,
    FRAMING_SAME_WEIGHTS_UNCORRELATED,
    FRAMING_STATED_ALWAYS_COOP,
    FRAMING_STATED_MATCHER,
    FRAMING_STATED_TRACK_RECORD,
    FRAMING_STATED_TRACK_RECORD_NOISY,
    FRAMING_TWIN,
    FRAMING_UNSTATED,
    LABEL_PRINT_ORDERS,
    SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    STATED_MATCHER_CLAUSE,
    STATED_TRACK_RECORD_CLAUSE,
    STATED_TRACK_RECORD_NOISY_CLAUSE,
    TEMPTATION_DOSE_GAME_ID,
    TEMPTATION_DOSE_PAYOFF_VARIANTS,
    TWIN_COUNTERPART_CLAUSE,
    assert_no_loaded_vocabulary,
    generate_counterpart_clause_prompt_rows,
    generate_framing_prompt_rows,
    generate_prompt_rows,
)
from games.tests.test_framing_stimulus import DEPENDENT, write_framings_file
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# What `games.evals` renders the sweep games under; framing rows take the same grading column the
# game-behavior section records for them, and both sweep games carry this one.
SWEEP_GRADING = "group-mix"

COUNTERPART_PARAGRAPH_OPENING = "About the other side:"


def _sweep_rows() -> list[tuple[str, str, dict[str, Any]]]:
    """Every row the full sweep renders, tagged with its game and framing."""
    return [
        (game_id, framing_id, row)
        for game_id in FRAMING_SWEEP_GAME_IDS
        for framing_id in COUNTERPART_FRAMING_IDS
        for order in LABEL_PRINT_ORDERS
        for row in generate_framing_prompt_rows(
            game_id, SWEEP_GRADING, framing_id=framing_id, split=SPLIT_EVAL, label_print_order=order
        )
    ]


class TestFramingRowsRenderCleanAndDistinct:
    def test_every_framing_renders_every_game_under_both_orders(self) -> None:
        """The full grid renders (the vocabulary guard runs inside the renderer), at known counts."""
        rows = _sweep_rows()
        per_game = dict.fromkeys(FRAMING_SWEEP_GAME_IDS, 0)
        for game_id, _, _ in rows:
            per_game[game_id] += 1
        # 4 eval-only frames x payoff variants x 2 coop mappings x 2 orders x every framing.
        assert per_game["twin-pd"] == 4 * 2 * 2 * 2 * len(COUNTERPART_FRAMING_IDS)
        assert per_game["public-goods"] == 4 * 1 * 2 * 2 * len(COUNTERPART_FRAMING_IDS)

    def test_prompt_ids_carry_their_framing_and_never_collide(self) -> None:
        rows = _sweep_rows()
        for _, framing_id, row in rows:
            assert f"--framing-{framing_id}" in row["prompt_id"]
        prompt_ids = [row["prompt_id"] for _, _, row in rows]
        assert len(prompt_ids) == len(set(prompt_ids))

    def test_the_twin_framing_reproduces_the_trained_rendering_byte_for_byte(self) -> None:
        """The reference cell must read exactly what the trained arms and the battery read."""
        for game_id in FRAMING_SWEEP_GAME_IDS:
            for order in LABEL_PRINT_ORDERS:
                framed = generate_framing_prompt_rows(
                    game_id,
                    SWEEP_GRADING,
                    framing_id=FRAMING_TWIN,
                    split=SPLIT_EVAL,
                    label_print_order=order,
                )
                trained = generate_prompt_rows(
                    game_id, SWEEP_GRADING, split=SPLIT_EVAL, label_print_order=order
                )
                assert [row["prompt"] for row in framed] == [row["prompt"] for row in trained]

    def test_the_unstated_framing_omits_the_paragraph_every_other_framing_carries(self) -> None:
        for game_id, framing_id, row in _sweep_rows():
            del game_id
            has_paragraph = COUNTERPART_PARAGRAPH_OPENING in row["prompt"]
            assert has_paragraph == (framing_id != FRAMING_UNSTATED)

    def test_the_disclosed_decision_names_the_rows_own_cooperative_label(self) -> None:
        """A disclosure naming the other label would probe exploitation of nothing nameable."""
        for _, framing_id, row in _sweep_rows():
            if framing_id != FRAMING_STATED_ALWAYS_COOP:
                continue
            assert f"they {row['coop_label']}." in row["prompt"]

    def test_the_matcher_clause_rides_its_rows_verbatim(self) -> None:
        for _, framing_id, row in _sweep_rows():
            if framing_id != FRAMING_STATED_MATCHER:
                continue
            assert STATED_MATCHER_CLAUSE in row["prompt"]

    def test_the_same_weights_framing_is_registered_and_rides_its_rows_verbatim(self) -> None:
        assert FRAMING_SAME_WEIGHTS_UNCORRELATED in COUNTERPART_FRAMING_IDS
        assert (
            COUNTERPART_FRAMINGS[FRAMING_SAME_WEIGHTS_UNCORRELATED]
            == SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE
        )
        rendered = [
            row
            for _, framing_id, row in _sweep_rows()
            if framing_id == FRAMING_SAME_WEIGHTS_UNCORRELATED
        ]
        assert rendered
        for row in rendered:
            assert SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE in row["prompt"]

    def test_the_same_weights_cell_asserts_twins_identity_and_denies_only_its_route(self) -> None:
        """What separates this cell from twin has to be the denied decision route and nothing else.
        If it reworded the shared-weights half, a movement between the two cells would read as a
        wording difference and the weights-against-coupling square would measure nothing."""
        route_marker = "and deciding on it "
        assert route_marker in TWIN_COUNTERPART_CLAUSE
        assert route_marker in SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE
        twin_identity, _, twin_route = TWIN_COUNTERPART_CLAUSE.partition(route_marker)
        identity, _, route = SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE.partition(route_marker)
        assert identity == twin_identity
        assert route != twin_route
        assert "need not come out the same as yours" in route

    def test_the_same_weights_clause_denies_the_route_without_naming_the_literature(self) -> None:
        """The denial lives in the mechanics because the guard refuses the shorter wording an
        author would reach for first, and the sabotage case is kept permanently rather than run
        once: a guard nobody has watched refuse anything is a reassuring message."""
        assert_no_loaded_vocabulary(SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE)
        names_the_literature = SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE.replace(
            "Its decision need not",
            "It is not your twin, so its decision need not",
        )
        with pytest.raises(ValueError, match="loaded vocabulary"):
            assert_no_loaded_vocabulary(names_the_literature)

    def test_the_track_record_clauses_extend_the_different_ai_clause_by_one_sentence(self) -> None:
        """The confirmation leg reads against the different-ai floor, so the pair must differ
        from that framing by exactly the appended track-record sentence -- and the noisy rung
        must differ from the certain one by exactly the word "almost"."""
        for clause in (STATED_TRACK_RECORD_CLAUSE, STATED_TRACK_RECORD_NOISY_CLAUSE):
            assert clause.startswith(DIFFERENT_AI_COUNTERPART_CLAUSE)
            appended = clause.removeprefix(DIFFERENT_AI_COUNTERPART_CLAUSE)
            assert appended.count(".") == 1
        assert STATED_TRACK_RECORD_NOISY_CLAUSE.replace("In almost every", "In every") == (
            STATED_TRACK_RECORD_CLAUSE
        )

    @pytest.mark.parametrize(
        ("framing_id", "clause", "floor_framing_id"),
        [
            (FRAMING_STATED_TRACK_RECORD, STATED_TRACK_RECORD_CLAUSE, FRAMING_DIFFERENT_AI),
            (
                FRAMING_STATED_TRACK_RECORD_NOISY,
                STATED_TRACK_RECORD_NOISY_CLAUSE,
                FRAMING_DIFFERENT_AI,
            ),
            (
                FRAMING_SAME_WEIGHTS_UNCORRELATED,
                SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
                FRAMING_TWIN,
            ),
            (
                FRAMING_SAME_WEIGHTS_UNCORRELATED,
                SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
                FRAMING_DIFFERENT_AI,
            ),
        ],
    )
    def test_a_framing_row_differs_from_its_floor_row_by_only_the_counterpart_line(
        self, framing_id: str, clause: str, floor_framing_id: str
    ) -> None:
        """Each of these framings is read against a named neighbour rather than against the
        grand mean, so the two renderings must differ by the counterpart paragraph and nothing
        else -- the track-record pair against the different-ai floor whose correlation sentence
        they append, and the same-weights-uncorrelated cell against both twin (which it differs
        from only by denying the shared decision route) and different-ai (only by asserting the
        shared weights)."""
        for game_id in FRAMING_SWEEP_GAME_IDS:
            framed = generate_framing_prompt_rows(
                game_id, SWEEP_GRADING, framing_id=framing_id, split=SPLIT_EVAL
            )
            floor = generate_framing_prompt_rows(
                game_id, SWEEP_GRADING, framing_id=floor_framing_id, split=SPLIT_EVAL
            )
            for framed_row, floor_row in zip(framed, floor, strict=True):
                assert clause in framed_row["prompt"]
                framed_lines = str(framed_row["prompt"]).splitlines()
                floor_lines = str(floor_row["prompt"]).splitlines()
                differing = [
                    pair
                    for pair in zip(framed_lines, floor_lines, strict=True)
                    if pair[0] != pair[1]
                ]
                assert len(differing) == 1
                assert differing[0][0].startswith("About the other side")


class TestFramingRowsRefuseWhatWouldUnmakeTheMeasurement:
    def test_the_training_split_is_refused(self) -> None:
        with pytest.raises(ValueError, match="measurement-only"):
            generate_framing_prompt_rows(
                "twin-pd", SWEEP_GRADING, framing_id=FRAMING_TWIN, split=SPLIT_TRAIN
            )

    def test_an_unknown_framing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown framing_id"):
            generate_framing_prompt_rows(
                "twin-pd", SWEEP_GRADING, framing_id="telepathic", split=SPLIT_EVAL
            )

    def test_a_game_without_a_counterpart_paragraph_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot take a counterpart framing"):
            generate_framing_prompt_rows(
                "dictator", SWEEP_GRADING, framing_id=FRAMING_TWIN, split=SPLIT_EVAL
            )

    def test_the_config_refuses_unknown_and_repeated_framings(self) -> None:
        with pytest.raises(ValueError, match="Unknown counterpart framings"):
            EvalConfig(counterpart_framings=("telepathic",))
        with pytest.raises(ValueError, match="more than once"):
            EvalConfig(counterpart_framings=(FRAMING_TWIN, FRAMING_TWIN))

    def test_the_battery_refuses_the_section_without_framings_before_generating(
        self, tmp_path: Path
    ) -> None:
        backend = MockBackend(responses=["reasoning</think>nothing"])
        with pytest.raises(ValueError, match="needs counterpart_framings"):
            run_eval_battery(
                backend,
                sections=(SECTION_FRAMING_SWEEP,),
                out_path=tmp_path / "step-0.jsonl",
                meta={"arm": "unit", "step": 0},
            )


def _framing_cooperative_backend() -> MockBackend:
    """Play the cooperative action on every sweep prompt, whichever label carries it."""
    coop_by_prompt = {row["prompt"]: row["coop_label"] for _, _, row in _sweep_rows()}

    def respond(prompt: str) -> str:
        label = coop_by_prompt.get(prompt)
        if label is None:
            return "reasoning</think>no such prompt was generated"
        return f"reasoning</think><action>{label}</action>"

    return MockBackend(responses=respond, model_id="mock-framing-cooperator")


class TestTheFramingSweepSectionEndToEnd:
    N_SAMPLES = 2

    def run_sweep(self, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
        out_path = tmp_path / "step-0.jsonl"
        summary = run_eval_battery(
            _framing_cooperative_backend(),
            sections=(SECTION_FRAMING_SWEEP,),
            out_path=out_path,
            meta={"arm": "unit", "step": 0},
            config=EvalConfig(
                counterpart_framings=COUNTERPART_FRAMING_IDS,
                label_print_orders=LABEL_PRINT_ORDERS,
                game_behavior_samples=self.N_SAMPLES,
            ),
        )
        return out_path, summary

    def test_every_record_carries_its_framing_and_the_sections_own_record_kind(
        self, tmp_path: Path
    ) -> None:
        out_path, _ = self.run_sweep(tmp_path)
        records = [json.loads(line) for line in out_path.read_text().splitlines()]
        assert records[0]["record"] == RECORD_META
        assert "frame_label_audit" in records[0]
        assert records[0]["eval_config"]["counterpart_framings"] == list(COUNTERPART_FRAMING_IDS)
        body = records[1:]
        assert len(body) == len(_sweep_rows()) * self.N_SAMPLES
        for record in body:
            assert record["record"] == SECTION_FRAMING_SWEEP
            assert record["counterpart_framing"] in COUNTERPART_FRAMINGS
            assert f"--framing-{record['counterpart_framing']}" in record["prompt_id"]

    def test_the_sweep_games_can_be_overridden_to_the_dose_ladder(self, tmp_path: Path) -> None:
        """The same-p-different-optimum cell: the temptation-dose ladder under dose framings.

        At one stated rate the ladder's rungs sit on BOTH sides of their own crossovers (p75:
        temptation-1.2/2 cooperate-optimal, temptation-5/10/20 defect-optimal), so this cell reads
        whether a policy combines the table with the rate or thresholds on the rate alone --
        pre-registration 7 of the track-record-v2 arm. The override exists because the default
        sweep games are pinned for every banked comparison; naming games is explicit opt-in.
        """
        out_path = tmp_path / "step-0.jsonl"
        framings = ("stated-track-record-p75", "stated-track-record-p90")
        summary = run_eval_battery(
            _framing_cooperative_backend(),
            sections=(SECTION_FRAMING_SWEEP,),
            out_path=out_path,
            meta={"arm": "unit", "step": 0},
            config=EvalConfig(
                counterpart_framings=framings,
                framing_sweep_games=(TEMPTATION_DOSE_GAME_ID,),
                game_behavior_samples=1,
            ),
        )
        records = [json.loads(line) for line in out_path.read_text().splitlines()]
        body = records[1:]
        assert body
        assert {record["game_id"] for record in body} == {TEMPTATION_DOSE_GAME_ID}
        assert {record["payoff_variant"] for record in body} == set(TEMPTATION_DOSE_PAYOFF_VARIANTS)
        rates = summary[SECTION_FRAMING_SWEEP]["coop_rate_by_game_framing"]
        assert set(rates) == {f"{TEMPTATION_DOSE_GAME_ID}::{framing_id}" for framing_id in framings}
        assert records[0]["eval_config"]["framing_sweep_games"] == [TEMPTATION_DOSE_GAME_ID]

    def test_an_unframeable_sweep_game_is_refused(self) -> None:
        """A game without a counterpart paragraph cannot be swept; the config refuses it."""
        with pytest.raises(ValueError, match="framing_sweep_games"):
            EvalConfig(counterpart_framings=(FRAMING_TWIN,), framing_sweep_games=("dictator",))

    def test_a_scripted_cooperator_reads_as_one_in_every_framing_cell(self, tmp_path: Path) -> None:
        """The sweep's own reduction reports what the backend did, per (game, framing), with
        denominators."""
        _, summary = self.run_sweep(tmp_path)
        rates = summary[SECTION_FRAMING_SWEEP]["coop_rate_by_game_framing"]
        expected_rows_per_framing = {"twin-pd": 4 * 2 * 2 * 2, "public-goods": 4 * 1 * 2 * 2}
        expected_keys = {
            f"{game_id}::{framing_id}"
            for game_id in FRAMING_SWEEP_GAME_IDS
            for framing_id in COUNTERPART_FRAMING_IDS
        }
        assert set(rates) == expected_keys
        for key, cell in rates.items():
            game_id = key.split("::")[0]
            assert cell["rate"] == 1.0
            assert cell["n_asked"] == expected_rows_per_framing[game_id]
            assert cell["n_parsed"] == cell["n_asked"]
            assert cell["n_records"] == cell["n_asked"] * self.N_SAMPLES


RUNTIME_SWEEP_GAME = "twin-pd"


def _runtime_rows(
    loaded: RuntimeFramings, framing_ids: Sequence[str]
) -> list[tuple[str, dict[str, Any]]]:
    """Every row the sweep renders for one game under the named framings, tagged with the framing."""
    return [
        (framing_id, row)
        for framing_id in framing_ids
        for row in generate_counterpart_clause_prompt_rows(
            RUNTIME_SWEEP_GAME,
            SWEEP_GRADING,
            clause=resolve_framing_clause(framing_id, loaded),
            framing_label=framing_id,
            split=SPLIT_EVAL,
        )
    ]


class TestTheSweepUnderRuntimeFramings:
    """Framings a wave authored outside version control, resolved second and rendered the same way.

    The registry answers first, so a loaded file can add framings and never re-state one; every
    property pinned here is one a reader of the trace assumes without being able to re-read the
    clause, which is why the cell identity has to move with the file's digest.
    """

    FRAMINGS = (FRAMING_TWIN, DEPENDENT)
    N_SAMPLES = 2

    def loaded(self, tmp_path: Path) -> RuntimeFramings:
        return load_framings(write_framings_file(tmp_path / "framings.json"))

    def config(self, loaded: RuntimeFramings) -> EvalConfig:
        return EvalConfig(
            counterpart_framings=self.FRAMINGS,
            framing_sweep_games=(RUNTIME_SWEEP_GAME,),
            game_behavior_samples=self.N_SAMPLES,
            runtime_framings=loaded,
        )

    def test_the_config_refuses_a_runtime_framing_with_no_file_loaded(self, tmp_path: Path) -> None:
        """The refusal is what keeps a typo from reading as a framing the registry simply lacks."""
        with pytest.raises(ValueError, match="Unknown counterpart framings"):
            EvalConfig(counterpart_framings=self.FRAMINGS)
        assert self.config(self.loaded(tmp_path)).counterpart_framings == self.FRAMINGS

    def test_the_planned_prompt_ids_carry_the_runtime_framing(self, tmp_path: Path) -> None:
        loaded = self.loaded(tmp_path)
        plan = plan_battery((SECTION_FRAMING_SWEEP,), self.config(loaded))
        assert plan
        for framing_id in self.FRAMINGS:
            assert [
                request
                for request in plan
                if any(f"--framing-{framing_id}" in str(field) for field in request.identity)
            ]

    def test_a_runtime_framings_prompts_carry_its_clause_as_the_one_inserted_paragraph(
        self, tmp_path: Path
    ) -> None:
        """The runtime clause reaches the model as the counterpart paragraph, and nothing else moves."""
        loaded = self.loaded(tmp_path)
        clause = loaded.clauses[DEPENDENT]
        unstated = generate_counterpart_clause_prompt_rows(
            RUNTIME_SWEEP_GAME,
            SWEEP_GRADING,
            clause=None,
            framing_label=FRAMING_UNSTATED,
            split=SPLIT_EVAL,
        )
        framed = [row for _, row in _runtime_rows(loaded, (DEPENDENT,))]
        for framed_row, stem_row in zip(framed, unstated, strict=True):
            assert str(framed_row["prompt"]).count(clause) == 1
            stem_lines = str(stem_row["prompt"]).splitlines()
            inserted = [
                line for line in str(framed_row["prompt"]).splitlines() if line not in stem_lines
            ]
            assert inserted == [f"{COUNTERPART_PARAGRAPH_OPENING} {clause}"]

    def test_a_registered_framing_renders_the_same_with_a_file_loaded(self, tmp_path: Path) -> None:
        """A loaded file cannot shadow the registry, so the reference cell stays byte-identical."""
        loaded = self.loaded(tmp_path)
        framed = [row for _, row in _runtime_rows(loaded, (FRAMING_TWIN,))]
        trained = generate_prompt_rows(RUNTIME_SWEEP_GAME, SWEEP_GRADING, split=SPLIT_EVAL)
        assert [row["prompt"] for row in framed] == [row["prompt"] for row in trained]

    def test_the_battery_stamps_every_record_and_records_the_digest_not_the_clauses(
        self, tmp_path: Path
    ) -> None:
        loaded = self.loaded(tmp_path)
        rows = _runtime_rows(loaded, self.FRAMINGS)
        coop_by_prompt = {str(row["prompt"]): str(row["coop_label"]) for _, row in rows}

        def respond(prompt: str) -> str:
            label = coop_by_prompt.get(prompt)
            if label is None:
                return "reasoning</think>no such prompt was generated"
            return f"reasoning</think><action>{label}</action>"

        out_path = tmp_path / "step-0.jsonl"
        summary = run_eval_battery(
            MockBackend(responses=respond, model_id="mock-runtime-framings"),
            sections=(SECTION_FRAMING_SWEEP,),
            out_path=out_path,
            meta={"arm": "unit", "step": 0},
            config=self.config(loaded),
        )
        records = [json.loads(line) for line in out_path.read_text().splitlines()]
        meta, body = records[0], records[1:]
        assert meta["eval_config"]["framings_digest"] == loaded.digest
        assert meta["eval_config"]["framings_file"] == str(loaded.path)
        assert loaded.clauses[DEPENDENT] not in json.dumps(meta["eval_config"])
        assert len(body) == len(rows) * self.N_SAMPLES
        assert {record["counterpart_framing"] for record in body} == set(self.FRAMINGS)
        rates = summary[SECTION_FRAMING_SWEEP]["coop_rate_by_game_framing"]
        assert set(rates) == {f"{RUNTIME_SWEEP_GAME}::{framing_id}" for framing_id in self.FRAMINGS}
        for cell in rates.values():
            assert cell["rate"] == 1.0
            assert cell["n_parsed"] == cell["n_asked"]

    def test_a_runtime_framing_states_no_coupling_and_an_unloaded_one_is_not_answered(
        self, tmp_path: Path
    ) -> None:
        """What `games.trace_judge_rates` asks of a record's framing, for one off the registry."""
        loaded = self.loaded(tmp_path)
        assert framing_states_coupling(DEPENDENT, loaded) is False
        assert framing_states_coupling(FRAMING_TWIN, loaded) is True
        with pytest.raises(ValueError, match=DEPENDENT):
            framing_states_coupling(DEPENDENT, None)
