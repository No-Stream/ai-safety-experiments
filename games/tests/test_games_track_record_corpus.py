"""Pin the track-record corpus builder: honesty, stem identity, and the two-sided mixture audit.

Offline and CPU-only. The source rows come from `generate_prompt_rows` itself rather than a
fixture file, so these tests exercise the same byte-identity path the real build runs against the
S3 corpus. Three checks here are the arm's own gates, each tested in the direction it can fail:

*   the HONESTY invariant (the clause's printed percent is the reward's `stated_match_prob`),
    driven red by monkeypatching the clause template into a liar;
*   the STEM identity (each output prompt is its source stem plus exactly one counterpart
    paragraph), driven red by editing a source prompt;
*   the TWO-SIDED mixture audit (every stem keeps rungs on both sides of its own EV crossover),
    driven red by an all-high grid and by a payoff table whose EV gap never moves with p.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import pytest

from games import prompts as prompts_module
from games.arms import PAYOFF_VARIANT_COLUMN as REGISTRY_PAYOFF_VARIANT_COLUMN
from games.payoffs import (
    STATED_MATCH_PROB_UNSET,
    MatrixGameSpec,
    assert_prisoners_dilemma,
    stated_match_gap,
)
from games.prompts import (
    TRACK_RECORD_GAME_ID,
    TRACK_RECORD_V2_GAME_ID,
    format_points,
    generate_prompt_rows,
    quantize_payoff_to_display,
)
from games.rewards import GRADING_VS_STATED_MATCH
from games.track_record_corpus import (
    MIN_EV_MARGIN,
    PAYOFF_VARIANT_COLUMN,
    SOURCE_GAME_ID,
    TRACK_RECORD_TRAINING_PERCENTS,
    TRACK_RECORD_V2_GRID,
    TRACK_RECORD_V2_TARGET_MARGIN,
    build_track_record_rows,
    build_track_record_v2_rows,
)


@pytest.fixture(scope="module")
def source_rows() -> list[dict[str, Any]]:
    """One reskin's four source rows: both payoff variants, both label mappings."""
    rows = generate_prompt_rows(SOURCE_GAME_ID, "group-mix", split="train")
    picked = [row for row in rows if row["reskin_id"] == rows[0]["reskin_id"]]
    assert len(picked) == 4, "expected 2 variants x 2 label mappings for one reskin"
    return picked


class TestTheBuiltCorpus:
    def test_every_row_is_honest_about_its_percent(self, source_rows: list[dict[str, Any]]) -> None:
        """The percent printed in the prompt IS the reward column, on every built row."""
        rows, _ = build_track_record_rows(source_rows)
        assert rows, "the build produced no rows"
        for row in rows:
            percent = round(float(row["stated_match_prob"]) * 100)
            assert f"In about {percent}% of matches so far" in row["prompt"], row["prompt_id"]
            assert row["stated_match_prob"] == percent / 100

    def test_rows_carry_the_new_game_and_grading(self, source_rows: list[dict[str, Any]]) -> None:
        rows, _ = build_track_record_rows(source_rows)
        assert {row["game_id"] for row in rows} == {TRACK_RECORD_GAME_ID}
        assert {row["grading"] for row in rows} == {GRADING_VS_STATED_MATCH}
        assert all(row["stated_match_prob"] != STATED_MATCH_PROB_UNSET for row in rows)

    def test_each_prompt_is_its_stem_plus_one_counterpart_paragraph(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        by_source = {str(row["prompt_id"]): str(row["prompt"]) for row in source_rows}
        rows, _ = build_track_record_rows(source_rows)
        for row in rows:
            reskin, variant = str(row["reskin_id"]), str(row[PAYOFF_VARIANT_COLUMN])
            coop = str(row["prompt_id"]).rsplit("--", 1)[-1]
            source_prompt = by_source[f"{SOURCE_GAME_ID}--{reskin}--{variant}--{coop}"]
            sections = str(row["prompt"]).split("\n\n")
            counterpart = [s for s in sections if s.startswith("About the other side: ")]
            assert len(counterpart) == 1, row["prompt_id"]
            assert (
                "\n\n".join(s for s in sections if not s.startswith("About the other side: "))
                == source_prompt
            ), row["prompt_id"]

    def test_the_thin_temptation_ten_rung_is_dropped_and_recorded(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        """85% sits 0.019 under temptation-10's 0.867 crossover: below the 0.05 floor, dropped."""
        rows, audit = build_track_record_rows(source_rows)
        t10_percents = {
            round(float(row["stated_match_prob"]) * 100)
            for row in rows
            if row[PAYOFF_VARIANT_COLUMN] == "temptation-10"
        }
        assert 85 not in t10_percents
        assert t10_percents == {99, 95, 65, 40}
        dropped = {
            (drop["source_prompt_id"], drop["match_percent"]) for drop in audit["dropped_cells"]
        }
        assert all(percent == 85 for _, percent in dropped)
        assert len(dropped) == 2  # the reskin's two temptation-10 label mappings
        # temptation-2 keeps all five rungs: 85% clears its 0.714 crossover by 0.19.
        t2_percents = {
            round(float(row["stated_match_prob"]) * 100)
            for row in rows
            if row[PAYOFF_VARIANT_COLUMN] == "temptation-2"
        }
        assert t2_percents == set(TRACK_RECORD_TRAINING_PERCENTS)

    def test_the_audit_carries_the_crossovers_and_both_side_counts(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        _, audit = build_track_record_rows(source_rows)
        assert audit["crossover_by_variant"]["temptation-2"] == pytest.approx(5 / 7)
        assert audit["crossover_by_variant"]["temptation-10"] == pytest.approx(13 / 15)
        assert audit["coop_optimal_rows"] > 0
        assert audit["defect_optimal_rows"] > 0
        assert audit["coop_optimal_rows"] + audit["defect_optimal_rows"] == audit["n_rows"]
        assert audit["min_ev_margin"] == MIN_EV_MARGIN

    def test_prompt_ids_cannot_collide(self, source_rows: list[dict[str, Any]]) -> None:
        rows, _ = build_track_record_rows(source_rows)
        prompt_ids = [row["prompt_id"] for row in rows]
        assert len(prompt_ids) == len(set(prompt_ids))

    def test_the_variant_column_name_matches_the_registry(self) -> None:
        """The builder restates the column to keep its import path cheap; the strings must agree."""
        assert PAYOFF_VARIANT_COLUMN == REGISTRY_PAYOFF_VARIANT_COLUMN


class TestTheAuditRefusesBrokenInputs:
    def test_a_one_sided_grid_is_refused(self, source_rows: list[dict[str, Any]]) -> None:
        """All rungs above every crossover: the track record never changes the answer."""
        with pytest.raises(ValueError, match="keeps no rung"):
            build_track_record_rows(source_rows, percents=(99, 95))

    def test_a_rigged_flat_payoff_table_is_refused(self, source_rows: list[dict[str, Any]]) -> None:
        """CC - DD == CD - DC keeps the EV gap constant in p, so no mixture can express."""
        rigged = [dict(row) for row in source_rows]
        rigged[0]["payoff_cc"] = 0.6
        rigged[0]["payoff_dd"] = 0.4
        rigged[0]["payoff_cd"] = 0.5
        rigged[0]["payoff_dc"] = 0.3
        with pytest.raises(ValueError, match="no interior EV crossover"):
            build_track_record_rows(rigged)

    def test_an_edited_source_stem_is_refused(self, source_rows: list[dict[str, Any]]) -> None:
        """A source prompt that is not the current renderer's own text fails byte identity."""
        edited = [dict(row) for row in source_rows]
        edited[0]["prompt"] = str(edited[0]["prompt"]).replace("points", "credits", 1)
        with pytest.raises(ValueError, match="differs from the current renderer"):
            build_track_record_rows(edited)

    def test_a_source_row_from_another_game_is_refused(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        foreign = [dict(row) for row in source_rows]
        foreign[0]["prompt_id"] = "twin-pd--somewhere--temptation-2--coop0"
        with pytest.raises(ValueError, match="is not a 'pd-unstated' training stem"):
            build_track_record_rows(foreign)

    def test_a_lying_clause_template_fails_the_honesty_check(
        self, source_rows: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sabotage item for the honesty invariant, kept as a permanent test.

        A template printing a different number than the reward column is exactly the dishonesty
        the invariant exists to catch, and nothing else in the pipeline would notice: the corpus
        would build, train, and produce plausible curves while the model was told one rate and
        graded on another.
        """
        liar = prompts_module.STATED_TRACK_RECORD_PERCENT_CLAUSE_TEMPLATE.replace(
            "{match_percent}%", "90%"
        )
        monkeypatch.setattr(prompts_module, "STATED_TRACK_RECORD_PERCENT_CLAUSE_TEMPLATE", liar)
        with pytest.raises(ValueError, match="stated rate and the graded rate"):
            build_track_record_rows(source_rows)


class TestTheV2Grid:
    """The v2 chain grid: margin balance and the anti-shortcut audits, on the built rows.

    Everything here recomputes from the ROWS' own payoff columns and stated_match_prob -- the
    numbers the reward function reads -- never from the design constants, so a builder that
    wrote different cells than it audited would fail here.
    """

    def test_every_cell_realizes_the_target_margin_on_displayed_numbers(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        """|EV(C) - EV(D)| at the stated rate is the target margin, on the quantized cells."""
        rows, _ = build_track_record_v2_rows(source_rows)
        assert rows, "the v2 build produced no rows"
        for row in rows:
            spec = MatrixGameSpec(
                game_id=str(row["game_id"]),
                payoff_cc=float(row["payoff_cc"]),
                payoff_cd=float(row["payoff_cd"]),
                payoff_dc=float(row["payoff_dc"]),
                payoff_dd=float(row["payoff_dd"]),
            )
            margin = stated_match_gap(spec, float(row["stated_match_prob"]))
            assert abs(abs(margin) - TRACK_RECORD_V2_TARGET_MARGIN) <= 5e-4, row["prompt_id"]
            assert_prisoners_dilemma(spec)

    def test_payoff_columns_sit_on_the_display_grid(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        """The reward columns ARE displayable numbers: quantizing them again is a no-op."""
        rows, _ = build_track_record_v2_rows(source_rows)
        for row in rows:
            for column in ("payoff_cc", "payoff_cd", "payoff_dc", "payoff_dd"):
                value = float(row[column])
                assert value == quantize_payoff_to_display(value), (row["prompt_id"], column)

    def test_the_prompt_displays_exactly_the_reward_columns(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        """Every point value printed in the outcome block is a reward column, verbatim.

        The v1 gap this closes: v1 displayed rounded points over unrounded internal cells, so
        the model's arithmetic and the reward's could disagree in the fourth decimal.
        """
        rows, _ = build_track_record_v2_rows(source_rows)
        for row in rows:
            expected = {
                format_points(float(row[column]))
                for column in ("payoff_cc", "payoff_cd", "payoff_dc", "payoff_dd")
            }
            credited = set(re.findall(r"you are credited (\S+) points", str(row["prompt"])))
            assert credited == expected, row["prompt_id"]

    def test_each_prompt_is_the_rescaled_stem_plus_one_paragraph(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        """Check (a): deleting the counterpart paragraph reproduces the rescaled-spec stem, and
        that stem differs from the banked original stem ONLY in the outcome block."""
        by_key = {
            (str(row["reskin_id"]), str(row["coop_label"])): str(row["prompt"])
            for row in source_rows
        }
        rows, _ = build_track_record_v2_rows(source_rows)
        for row in rows:
            sections = str(row["prompt"]).split("\n\n")
            counterpart = [s for s in sections if s.startswith("About the other side: ")]
            assert len(counterpart) == 1, row["prompt_id"]
            stem_sections = [s for s in sections if not s.startswith("About the other side: ")]
            original = by_key[(str(row["reskin_id"]), str(row["coop_label"]))]
            original_sections = original.split("\n\n")
            assert len(stem_sections) == len(original_sections), row["prompt_id"]
            differing = [
                index
                for index, (ours, theirs) in enumerate(
                    zip(stem_sections, original_sections, strict=True)
                )
                if ours != theirs
            ]
            assert len(differing) == 1, (row["prompt_id"], differing)
            index = differing[0]
            assert "credited" in stem_sections[index], row["prompt_id"]
            skeleton = re.sub(r"credited \S+ points", "credited N points", stem_sections[index])
            original_skeleton = re.sub(
                r"credited \S+ points", "credited N points", original_sections[index]
            )
            assert skeleton == original_skeleton, row["prompt_id"]

    def test_v2_rows_carry_the_v2_game_and_are_honest(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        rows, _ = build_track_record_v2_rows(source_rows)
        assert {row["game_id"] for row in rows} == {TRACK_RECORD_V2_GAME_ID}
        assert {row["grading"] for row in rows} == {GRADING_VS_STATED_MATCH}
        for row in rows:
            percent = round(float(row["stated_match_prob"]) * 100)
            assert f"In about {percent}% of matches so far" in row["prompt"], row["prompt_id"]
        prompt_ids = [row["prompt_id"] for row in rows]
        assert len(prompt_ids) == len(set(prompt_ids))

    def test_duplicate_stem_keys_across_source_variants_collapse(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        """The real source holds the same reskin under both payoff variants; the v2 render does
        not depend on the source variant, so those stems must collapse to one, not collide."""
        rows, audit = build_track_record_v2_rows(source_rows)
        stem_keys = {
            (str(row["reskin_id"]), str(row["coop_label"]), str(row["label_print_order"]))
            for row in source_rows
        }
        assert audit["n_stems"] == len(stem_keys)
        assert len(rows) == len({row["prompt_id"] for row in rows})

    def test_the_balance_audit_numbers(self, source_rows: list[dict[str, Any]]) -> None:
        _, audit = build_track_record_v2_rows(source_rows)
        assert audit["best_rate_lookup_accuracy"] <= 0.70
        assert 0.45 <= audit["coop_optimal_rows"] / audit["n_rows"] <= 0.55
        assert audit["anchor_row_mass"] <= 0.35
        for fraction in audit["coop_fraction_by_table"].values():
            assert 0.3 <= fraction <= 0.7
        assert audit["target_margin"] == TRACK_RECORD_V2_TARGET_MARGIN


class TestTheV2AuditsRefuseRiggedInputs:
    """Each v2 gate, driven red once: a gate never watched failing is not a gate."""

    def test_a_one_sided_grid_fails_the_table_balance_audit(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        """A grid whose every cell is defect-optimal: each table has no minority side."""
        rigged = tuple(replace(slot, coop_rung=None) for slot in TRACK_RECORD_V2_GRID)
        with pytest.raises(ValueError, match="one side"):
            build_track_record_v2_rows(source_rows, grid=rigged)

    def test_slot_private_rungs_fail_the_rate_lookup_audit(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        """The design v2 rejected, kept red: unshared rungs leave every table balanced while a
        6-value rate-memorization policy scores 1.0 -- exactly what the lookup gate exists for."""
        rigged = tuple(
            replace(slot, defect_rung=rung)
            for slot, rung in zip(TRACK_RECORD_V2_GRID, (51, 64, 80), strict=True)
        )
        with pytest.raises(ValueError, match="rate-lookup"):
            build_track_record_v2_rows(source_rows, grid=rigged)

    def test_a_cell_on_the_wrong_side_of_its_own_arithmetic_is_refused(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        """A high-crossover table planted in the low slot: both its rungs are defect-optimal,
        so the slot's claimed cooperate side contradicts the displayed table's own arithmetic."""
        low = TRACK_RECORD_V2_GRID[0]
        rigged = (
            replace(low, table_variants=(*low.table_variants, "xover89-floor15")),
            *TRACK_RECORD_V2_GRID[1:],
        )
        with pytest.raises(ValueError, match="grid places this cell"):
            build_track_record_v2_rows(source_rows, grid=rigged)

    def test_a_margin_starved_cell_is_refused(self, source_rows: list[dict[str, Any]]) -> None:
        """A rung too close to its slot's crossover needs k > 1, which the payoff box forbids."""
        rigged = tuple(
            replace(slot, defect_rung=68) if slot.name == "mid" else slot
            for slot in TRACK_RECORD_V2_GRID
        )
        with pytest.raises(ValueError, match="needs scale k > 1"):
            build_track_record_v2_rows(source_rows, grid=rigged)

    def test_a_lying_clause_template_fails_v2_honesty(
        self, source_rows: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        liar = prompts_module.STATED_TRACK_RECORD_PERCENT_CLAUSE_TEMPLATE.replace(
            "{match_percent}%", "90%"
        )
        monkeypatch.setattr(prompts_module, "STATED_TRACK_RECORD_PERCENT_CLAUSE_TEMPLATE", liar)
        with pytest.raises(ValueError, match="stated rate and the graded rate"):
            build_track_record_v2_rows(source_rows)

    def test_an_edited_source_stem_is_refused_by_v2_too(
        self, source_rows: list[dict[str, Any]]
    ) -> None:
        edited = [dict(row) for row in source_rows]
        edited[0]["prompt"] = str(edited[0]["prompt"]).replace("points", "credits", 1)
        with pytest.raises(ValueError, match="differs from the current renderer"):
            build_track_record_v2_rows(edited)
