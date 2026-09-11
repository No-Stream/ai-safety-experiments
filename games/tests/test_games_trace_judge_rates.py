"""The trace judge's rates and calibration: binomial bounds, clustered bands, per-scope rates and
deltas, the drawn overlay, and hand-label validation with its sabotage case.

Fixtures are shared with ``test_games_trace_judge`` (synthetic traces only; this file is tracked).
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from games.tests.test_games_trace_judge import (
    COMPLETION,
    SCOPE,
    VERDICT_PAYLOAD,
    VISIBLE,
    make_record,
    write_jsonl,
)
from games.trace_judge_rates import (
    DECISION_COOPERATE,
    DECISION_DEFECT,
    DOMINANCE_CAVEAT,
    FRAMINGS_DIGEST_FIELD,
    binomial_bounds,
    clustered_2se,
    coupling_clause_in_prompt,
    decision_of,
    hand_label_sample_files,
    load_hand_labels,
    load_sample_handles,
    resolve_hand_keys,
    sample_row_key,
    scope_label,
    trace_rates,
    validation_report,
)
from games.trace_judge_schema import RUBRIC_VERSION, keyword_basis, rubric_digest

if TYPE_CHECKING:
    from collections.abc import Iterable

    from games.trace_judge_rates import Handle
    from games.trace_judge_records import TraceRecord

RUNTIME_FRAMING = "dependent"
"""A framing id no tracked registry carries; the prosocial-breadth wave loads it at runtime."""

CELL_0 = f"arm-x|cell-y|{SCOPE}@0"
CELL_70 = f"arm-x|cell-y|{SCOPE}@70"
GOLDEN_RATES = Path(__file__).resolve().parent / "data" / "trace_judge_rates_cooperators_only.json"


def judged_row(record: TraceRecord, **verdict_overrides: Any) -> dict[str, Any]:
    verdict = dict(VERDICT_PAYLOAD)
    verdict.update(verdict_overrides)
    return {
        "key": record.key,
        "arm": record.arm,
        "cell": record.cell,
        "step": record.step,
        "section": record.section,
        "game_id": record.game_id,
        "counterpart_framing": record.counterpart_framing,
        "payoff_variant": record.payoff_variant,
        "decision": record.decision,
        "prompt_id": record.prompt_id,
        "sample_index": record.sample_index,
        "completion_sha256": record.completion_sha256,
        "keyword_basis": keyword_basis(record.completion),
        "rubric_version": RUBRIC_VERSION,
        "rubric_digest": rubric_digest(),
        "evidence_not_found": False,
        "verdict": verdict,
    }


def between_prompt_2se(shares: list[float]) -> float:
    return 2 * statistics.stdev(shares) / math.sqrt(len(shares))


def census_rows(records: Iterable[TraceRecord]) -> list[dict[str, Any]]:
    return [
        {
            "key": r.key,
            "arm": r.arm,
            "cell": r.cell,
            "step": r.step,
            "section": r.section,
            "game_id": r.game_id,
            "counterpart_framing": r.counterpart_framing,
            "payoff_variant": r.payoff_variant,
            "decision": r.decision,
            "prompt_id": r.prompt_id,
            "sample_index": r.sample_index,
            FRAMINGS_DIGEST_FIELD: r.framings_digest,
            "judgeable": bool(r.completion.strip()),
        }
        for r in records
    ]


def cooperators_only_fixture() -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]], set[Handle]
]:
    """A two-step cooperators-only census with every denominator, a one-step scope and a drawn overlay.

    The golden file at ``GOLDEN_RATES`` was rendered from this fixture by the rates code as it stood
    before the decision and per-render selectors landed (2026-09-03). A cooperators-only census has to
    render byte-identically under every later revision: every wave-3 rates file was read under those
    labels and none of them may move.
    """
    step0 = [
        make_record(step=0, prompt_id=f"p{p}", sample_index=s) for p in range(3) for s in range(2)
    ]
    step70 = [
        make_record(step=70, prompt_id=f"p{p}", sample_index=s) for p in range(3) for s in range(2)
    ]
    third_of_p0 = make_record(step=0, prompt_id="p0", sample_index=2)
    skipped = make_record(step=0, prompt_id="p9", sample_index=0, completion="", visible_text="")
    errored = make_record(step=0, prompt_id="p9", sample_index=1)
    unjudged = make_record(step=0, prompt_id="p9", sample_index=2)
    one_step_scope = make_record(
        step=0,
        section="game-behavior",
        game_id="pd-reskin",
        prompt_id="pd-reskin--other-frame--temptation-2--coop0",
        counterpart_framing=None,
    )
    census = census_rows([*step0, *step70, third_of_p0, skipped, errored, unjudged, one_step_scope])
    step0_verdicts: list[dict[str, Any]] = [
        {"decision_basis": "mirror", "counterpart_assumption": "copies-me"},
        {
            "decision_basis": "dominance",
            "counterpart_assumption": "independent",
            "diagonal_comparison": "none",
            "dominance_rejected": False,
        },
        {
            "decision_basis": "prosocial-other-regarding",
            "secondary_basis": "benchmark-known-answer",
            "counterpart_assumption": "not-discussed",
            "diagonal_comparison": "supporting",
        },
        {"decision_basis": "convergence", "counterpart_assumption": "correlated"},
        {"decision_basis": "mirror", "counterpart_assumption": "copies-me"},
        {
            "decision_basis": "instruction-compliance",
            "counterpart_assumption": "disclosed-and-matched",
            "diagonal_comparison": "none",
            "dominance_raised": False,
            "dominance_rejected": False,
            "payoff_reasoning_present": False,
        },
    ]
    step70_assumptions = {("p0", 0): "copies-me", ("p0", 1): "correlated", ("p1", 0): "copies-me"}
    judged = {
        record.key: judged_row(record, **verdict)
        for record, verdict in zip(step0, step0_verdicts, strict=True)
    }
    judged[third_of_p0.key] = judged_row(third_of_p0)
    for record in step70:
        judged[record.key] = judged_row(
            record,
            counterpart_assumption=step70_assumptions.get(
                (record.prompt_id, record.sample_index), "independent"
            ),
        )
    judged[one_step_scope.key] = judged_row(one_step_scope)
    census_by_key = {str(row["key"]): row for row in census}
    judged[errored.key] = {**census_by_key[errored.key], "judge_error": "boom"}
    drawn = {record.handle for record in step0[:2]} | {(0, "never-judged", 0, "0" * 64)}
    return judged, census, drawn


class TestBinomialBounds:
    def test_zero_count_matches_the_absence_floor_convention(self) -> None:
        bounds = binomial_bounds(0, 40)
        assert bounds["upper_95_one_sided"] == pytest.approx(1 - 0.05 ** (1 / 40))
        assert bounds["ci95_lower"] == 0.0
        assert bounds["ci95_upper"] == pytest.approx(1 - 0.025 ** (1 / 40))
        assert binomial_bounds(0, 4)["upper_95_one_sided"] == pytest.approx(0.527, abs=1e-3)

    def test_interior_count_brackets_the_share(self) -> None:
        bounds = binomial_bounds(3, 10)
        assert bounds["ci95_lower"] == pytest.approx(0.0667, abs=1e-3)
        assert bounds["ci95_upper"] == pytest.approx(0.6525, abs=1e-3)

    def test_full_count_has_upper_one(self) -> None:
        assert binomial_bounds(5, 5)["ci95_upper"] == 1.0
        assert binomial_bounds(5, 5)["upper_95_one_sided"] == 1.0

    def test_empty_denominator_is_none(self) -> None:
        assert binomial_bounds(0, 0) == {
            "ci95_lower": None,
            "ci95_upper": None,
            "upper_95_one_sided": None,
        }


class TestClustered2SE:
    def test_matches_the_hand_computation(self) -> None:
        """p-hat = 3/6; cluster residual sums +1.0 and -1.0; V = (2/1) * 2 / 36; SE = 1/3."""
        clusters = {"a": [1.0, 1.0, 1.0, 0.0], "b": [0.0, 0.0]}
        assert clustered_2se(clusters) == pytest.approx(2 / 3)

    def test_one_cluster_is_none(self) -> None:
        assert clustered_2se({"a": [1.0, 0.0]}) is None


class TestTraceRates:
    def records(self) -> tuple[list[TraceRecord], list[TraceRecord]]:
        step0 = [
            make_record(step=0, prompt_id=f"p{p}", sample_index=s)
            for p in range(3)
            for s in range(2)
        ]
        step70 = [
            make_record(step=70, prompt_id=f"p{p}", sample_index=s)
            for p in range(3)
            for s in range(2)
        ]
        return step0, step70

    def test_cell_counts_denominators_shares_and_bounds(self) -> None:
        """p0 has three judged cooperators and p1, p2 two each, so the prompt-first mean and the pooled
        share differ (coupling 0.5556 vs 0.5714) and the assertions below cannot pass by coincidence."""
        step0, _ = self.records()
        third_of_p0 = make_record(step=0, prompt_id="p0", sample_index=2)
        skipped = make_record(
            step=0, prompt_id="p9", sample_index=0, completion="", visible_text=""
        )
        errored = make_record(step=0, prompt_id="p9", sample_index=1)
        unjudged = make_record(step=0, prompt_id="p9", sample_index=2)
        census = census_rows([*step0, third_of_p0, skipped, errored, unjudged])
        judged = {
            third_of_p0.key: judged_row(
                third_of_p0, decision_basis="mirror", counterpart_assumption="copies-me"
            ),
            step0[0].key: judged_row(
                step0[0], decision_basis="mirror", counterpart_assumption="copies-me"
            ),
            step0[1].key: judged_row(
                step0[1],
                decision_basis="dominance",
                counterpart_assumption="independent",
                diagonal_comparison="none",
                dominance_rejected=False,
            ),
            step0[2].key: judged_row(
                step0[2],
                decision_basis="prosocial-other-regarding",
                secondary_basis="benchmark-known-answer",
                counterpart_assumption="not-discussed",
                diagonal_comparison="supporting",
            ),
            step0[3].key: judged_row(
                step0[3], decision_basis="convergence", counterpart_assumption="correlated"
            ),
            step0[4].key: judged_row(
                step0[4], decision_basis="mirror", counterpart_assumption="copies-me"
            ),
            step0[5].key: judged_row(
                step0[5],
                decision_basis="instruction-compliance",
                counterpart_assumption="disclosed-and-matched",
                diagonal_comparison="none",
                dominance_raised=False,
                dominance_rejected=False,
                payoff_reasoning_present=False,
            ),
            errored.key: {**census[8], "judge_error": "boom"},
        }
        rates = trace_rates(judged, census)
        cell = rates["cells"][CELL_0]
        assert cell["examined"] == 10
        assert cell["skipped_empty"] == 1
        assert cell["errored"] == 1
        assert cell["not_judged"] == 1
        assert cell["judged"] == 7
        assert cell["basis"]["mirror"] == 3
        assert cell["basis"]["instruction-compliance"] == 1
        assert cell["basis"]["ev-arithmetic"] == 0, "every level is present so a zero is a zero"
        assert cell["secondary_basis"]["benchmark-known-answer"] == 1
        assert cell["secondary_basis"]["null"] == 6
        assert cell["counterpart_assumption"]["disclosed-and-matched"] == 1
        assert cell["diagonal_comparison"] == {"none": 2, "supporting": 1, "only": 4}
        coupling = cell["coupling_story"]
        assert (coupling["k"], coupling["n"]) == (4, 7), (
            "a disclosed counterpart is not an invented story"
        )
        assert coupling["share"] == pytest.approx(4 / 7)
        assert coupling["upper_95_one_sided"] == pytest.approx(
            binomial_bounds(4, 7)["upper_95_one_sided"]
        )
        assert cell["diagonal_comparison:only"]["k"] == 4
        assert cell["diagonal_comparison:supporting"]["k"] == 1
        assert cell["dominance_raised"]["k"] == 6
        assert cell["dominance_rejected"]["k"] == 5
        assert cell["payoff_reasoning_present"]["k"] == 6
        # Per prompt: p0 = [copies-me, independent, copies-me] -> 2/3; p1 -> 1/2; p2 -> 1/2.
        assert cell["prompt_mean"]["coupling_story"] == pytest.approx((2 / 3 + 0.5 + 0.5) / 3)
        assert cell["prompt_mean"]["coupling_story"] != pytest.approx(coupling["share"])
        assert cell["prompt_mean"]["basis:mirror"] == pytest.approx((2 / 3 + 0.0 + 0.5) / 3)
        assert cell["n_prompts"] == 3
        assert cell["coupling_clause_in_prompt"] is False, "the unstated framing states nothing"

    def test_cells_are_keyed_by_scope_so_framings_and_variants_never_pool(self) -> None:
        """instruction-compliance was 7 of 7 in the disclosed-counterpart framing: a rates file keyed
        by the battery cell directory alone would average that framing into the undisclosed ones."""
        disclosed = make_record(
            prompt_id="twin-pd--frame--temptation-2--framing-stated-always-coop--coop0",
            counterpart_framing="stated-always-coop",
        )
        other_variant = make_record(
            prompt_id="twin-pd--frame--temptation-10--framing-unstated--coop0",
            payoff_variant="temptation-10",
        )
        no_framing = make_record(
            section="training-frames",
            game_id="pd-unstated",
            prompt_id="pd-unstated--frame--temptation-2--coop0",
            counterpart_framing=None,
        )
        records = [make_record(), disclosed, other_variant, no_framing]
        judged = {
            make_record().key: judged_row(make_record()),
            disclosed.key: judged_row(disclosed, decision_basis="instruction-compliance"),
            other_variant.key: judged_row(other_variant),
            no_framing.key: judged_row(no_framing),
        }
        rates = trace_rates(judged, census_rows(records))
        assert set(rates["cells"]) == {
            CELL_70,
            "arm-x|cell-y|framing-sweep::twin-pd::stated-always-coop::temptation-2@70",
            "arm-x|cell-y|framing-sweep::twin-pd::unstated::temptation-10@70",
            "arm-x|cell-y|training-frames::pd-unstated::no-framing::temptation-2@70",
        }
        assert rates["cells"][CELL_70]["basis"]["instruction-compliance"] == 0
        assert (
            scope_label(census_rows([no_framing])[0])
            == "training-frames::pd-unstated::no-framing::temptation-2"
        )

    def test_cells_are_keyed_by_decision_so_cooperators_and_defectors_never_pool(self) -> None:
        """A defector's scope carries ``::defect``; a cooperator's is the bare label every wave-3 rates
        file used, and its numbers are exactly what they would be with no defectors beside it."""
        cooperators = [
            make_record(step=step, prompt_id=f"p{p}", sample_index=0)
            for step in (0, 70)
            for p in range(3)
        ]
        defectors = [
            make_record(step=step, prompt_id=f"p{p}", sample_index=1, decision=DECISION_DEFECT)
            for step in (0, 70)
            for p in range(3)
        ]
        judged = {r.key: judged_row(r) for r in cooperators} | {
            r.key: judged_row(
                r,
                decision_basis="dominance",
                counterpart_assumption="independent",
                diagonal_comparison="none",
                dominance_rejected=False,
            )
            for r in defectors
        }
        rates = trace_rates(judged, census_rows([*cooperators, *defectors]))
        defect_0 = f"arm-x|cell-y|{SCOPE}::defect@0"
        defect_70 = f"arm-x|cell-y|{SCOPE}::defect@70"
        assert set(rates["cells"]) == {CELL_0, CELL_70, defect_0, defect_70}
        assert rates["cells"][CELL_0]["basis"] == {**rates["cells"][CELL_0]["basis"], "mirror": 3}
        assert rates["cells"][CELL_0]["basis"]["dominance"] == 0
        assert rates["cells"][defect_0]["basis"]["dominance"] == 3
        assert rates["cells"][defect_0]["basis"]["mirror"] == 0
        assert rates["cells"][defect_0]["coupling_story"]["k"] == 0
        assert set(rates["deltas"]) == {f"arm-x|cell-y|{SCOPE}", f"arm-x|cell-y|{SCOPE}::defect"}
        alone = trace_rates({r.key: judged_row(r) for r in cooperators}, census_rows(cooperators))
        for label in (CELL_0, CELL_70):
            assert rates["cells"][label] == alone["cells"][label]
            assert rates["prompts"][label] == alone["prompts"][label]
        assert rates["deltas"][f"arm-x|cell-y|{SCOPE}"] == alone["deltas"][f"arm-x|cell-y|{SCOPE}"]
        assert scope_label(census_rows([defectors[0]])[0]) == f"{SCOPE}::defect"
        assert scope_label(census_rows([cooperators[0]])[0]) == SCOPE

    def test_a_census_without_a_decision_stamp_reads_as_cooperators_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every census before 2026-09-03 came from a cooperators-only loader, so the absent stamp is
        ``cooperate``; the warning is there because the scratch driver's defector censuses lack it too."""
        record = make_record()
        unstamped = [
            {k: v for k, v in row.items() if k != "decision"} for row in census_rows([record])
        ]
        assert decision_of(unstamped[0]) == DECISION_COOPERATE
        with caplog.at_level("WARNING", logger="games.trace_judge_rates"):
            rates = trace_rates({record.key: judged_row(record)}, unstamped)
        assert set(rates["cells"]) == {CELL_70}
        assert "1 of 1 census rows carry no decision stamp" in caplog.text

    def test_a_judged_row_whose_decision_disagrees_with_its_census_row_is_refused(self) -> None:
        """Same key, another decision: the tree changed under the key and the verdict is about a record
        this census does not hold; pooling it would file a defector's reasoning under the cooperators."""
        record = make_record()
        turned = dict(judged_row(record), decision=DECISION_DEFECT)
        with pytest.raises(ValueError, match="stamped decision='defect'"):
            trace_rates({record.key: turned}, census_rows([record]))

    def test_a_stored_row_stamped_with_a_selector_rather_than_a_decision_is_refused(self) -> None:
        census = census_rows([make_record()])
        census[0]["decision"] = "all"
        with pytest.raises(ValueError, match="never a selector"):
            trace_rates({}, census)

    def test_each_cell_is_stamped_with_whether_its_prompt_stated_the_coupling(self) -> None:
        """The control's 'invented coupling story' reading applies only where the prompt said nothing
        about the counterpart's decision. A framing-sweep cell takes the stamp from its framing, since
        the sweep swaps the game's own paragraph out; every other section takes it from the game."""
        twin_framed = make_record(
            prompt_id="twin-pd--frame--temptation-2--framing-twin--coop0",
            counterpart_framing="twin",
        )
        unstated = make_record()
        game_behaviour_twin = make_record(
            section="game-behavior",
            prompt_id="twin-pd--frame--temptation-2--coop0",
            counterpart_framing=None,
        )
        game_behaviour_silent = make_record(
            section="game-behavior",
            game_id="pd-reskin",
            prompt_id="pd-reskin--other-frame--temptation-2--coop0",
            counterpart_framing=None,
        )
        records = [twin_framed, unstated, game_behaviour_twin, game_behaviour_silent]
        rates = trace_rates({r.key: judged_row(r) for r in records}, census_rows(records))
        stamps = {
            label: cell["coupling_clause_in_prompt"] for label, cell in rates["cells"].items()
        }
        assert stamps == {
            "arm-x|cell-y|framing-sweep::twin-pd::twin::temptation-2@70": True,
            CELL_70: False,
            "arm-x|cell-y|game-behavior::twin-pd::no-framing::temptation-2@70": True,
            "arm-x|cell-y|game-behavior::pd-reskin::no-framing::temptation-2@70": False,
        }

    def test_a_runtime_framing_reads_decoupled_when_the_cell_recorded_its_framings_file(
        self,
    ) -> None:
        """`--framings-file` renders framings the tracked registry does not carry, so the stamp
        cannot be looked up from a clause the reader never sees.

        `games.framing_stimulus.load_framings` refuses any runtime clause that asserts the coupling,
        so a cell whose meta records a framings file rendered decoupled framings by construction, and
        the digest that meta carries is what the census stamps.
        """
        record = make_record(
            counterpart_framing=RUNTIME_FRAMING,
            prompt_id=f"twin-pd--frame--temptation-2--framing-{RUNTIME_FRAMING}--coop0",
            framings_digest="0123456789abcdef",
        )
        assert coupling_clause_in_prompt(census_rows([record])[0]) is False
        rates = trace_rates({record.key: judged_row(record)}, census_rows([record]))
        assert [cell["coupling_clause_in_prompt"] for cell in rates["cells"].values()] == [False]

    def test_a_framing_off_the_registry_with_no_framings_file_is_refused_by_name(self) -> None:
        """The typo case, which must not read as decoupled: without a framings file in the cell's
        meta there is no clause anywhere that the id could have named, so the answer is unknown."""
        record = make_record(
            counterpart_framing=RUNTIME_FRAMING,
            prompt_id=f"twin-pd--frame--temptation-2--framing-{RUNTIME_FRAMING}--coop0",
        )
        assert record.framings_digest == ""
        with pytest.raises(ValueError, match=RUNTIME_FRAMING):
            coupling_clause_in_prompt(census_rows([record])[0])
        with pytest.raises(ValueError, match=RUNTIME_FRAMING):
            trace_rates({record.key: judged_row(record)}, census_rows([record]))

    def test_a_registered_framing_is_read_off_its_clause_whatever_the_meta_says(self) -> None:
        """A framings file cannot relabel a registered rung: resolution consults the registry first."""
        twin = make_record(
            counterpart_framing="twin",
            prompt_id="twin-pd--frame--temptation-2--framing-twin--coop0",
            framings_digest="0123456789abcdef",
        )
        assert coupling_clause_in_prompt(census_rows([twin])[0]) is True

    def test_a_row_judged_under_another_scaffold_digest_is_refused_not_pooled(self) -> None:
        record = make_record()
        stale = dict(judged_row(record), rubric_digest="0000000000000000")
        with pytest.raises(ValueError, match="digest"):
            trace_rates({record.key: stale}, census_rows([record]))

    def test_keyword_agreement_is_reported_not_merged(self) -> None:
        step0, _ = self.records()
        agree = judged_row(step0[0], decision_basis="mirror")  # completion matches MIRROR_RE
        disagree = judged_row(step0[1], decision_basis="dominance")
        rates = trace_rates({step0[0].key: agree, step0[1].key: disagree}, census_rows(step0[:2]))
        cell = rates["cells"][CELL_0]
        assert cell["keyword_agreement"] == {"agree": 1, "n": 2, "rate": 0.5}
        assert cell["keyword_disagreements"] == [
            {
                "key": step0[1].key,
                "judge": "dominance",
                "judge_coarse": "none-matched",
                "keyword": "mirror",
            }
        ]
        assert cell["basis"]["dominance"] == 1, "the judge's answer stands"

    def test_paired_between_prompt_delta_with_the_known_sd(self) -> None:
        step0, step70 = self.records()
        judged: dict[str, dict[str, Any]] = {}
        for record in step0:
            judged[record.key] = judged_row(record, counterpart_assumption="independent")
        # Per-prompt coupling shares at step 70: p0 = 1.0, p1 = 0.5, p2 = 0.0.
        assumptions = {("p0", 0): "copies-me", ("p0", 1): "correlated", ("p1", 0): "copies-me"}
        for record in step70:
            judged[record.key] = judged_row(
                record,
                counterpart_assumption=assumptions.get(
                    (record.prompt_id, record.sample_index), "independent"
                ),
            )
        rates = trace_rates(judged, census_rows([*step0, *step70]))
        arm_scope = f"arm-x|cell-y|{SCOPE}"
        delta = rates["deltas"][arm_scope]["dimensions"]["coupling_story"]
        assert rates["deltas"][arm_scope]["steps"] == [0, 70]
        assert delta["delta"] == pytest.approx(0.5)
        assert delta["band_method"] == "paired-between-prompt"
        assert delta["n_paired"] == 3
        assert delta["n_unpaired"] == 0
        # deltas [1.0, 0.5, 0.0]: SD 0.5, 2SE = 2 * 0.5 / sqrt(3).
        assert delta["paired_2se"] == pytest.approx(2 * 0.5 / math.sqrt(3))
        assert delta["clustered_2se"] is not None
        assert delta["level"] == {"0": 0.0, "70": pytest.approx(0.5)}

    def test_quadrature_fallback_says_so_when_pairing_is_unlicensed(self) -> None:
        """Disjoint prompt sets at the two steps, each with a nonzero between-prompt spread, so the
        quadrature band is a number the test can recompute rather than a zero anything would match."""
        step0 = [
            make_record(step=0, prompt_id=f"q{p}", sample_index=s)
            for p in range(3)
            for s in range(2)
        ]
        _, step70 = self.records()
        # Per-prompt coupling shares: step 0 -> q0 1.0, q1 0.5, q2 0.0; step 70 -> p0 1.0, p1 1.0, p2 0.0.
        assumptions = {
            ("q0", 0): "copies-me",
            ("q0", 1): "correlated",
            ("q1", 0): "copies-me",
            ("p0", 0): "copies-me",
            ("p0", 1): "copies-me",
            ("p1", 0): "correlated",
            ("p1", 1): "correlated",
        }
        judged = {
            r.key: judged_row(
                r,
                counterpart_assumption=assumptions.get(
                    (r.prompt_id, r.sample_index), "independent"
                ),
            )
            for r in [*step0, *step70]
        }
        rates = trace_rates(judged, census_rows([*step0, *step70]))
        delta = rates["deltas"][f"arm-x|cell-y|{SCOPE}"]["dimensions"]["coupling_story"]
        assert delta["n_paired"] == 0
        assert delta["n_unpaired"] == 6
        assert delta["band_method"] == "quadrature-unpaired"
        assert delta["paired_2se"] is None
        low_2se = between_prompt_2se([1.0, 0.5, 0.0])
        high_2se = between_prompt_2se([1.0, 1.0, 0.0])
        assert low_2se > 0
        assert high_2se > 0
        assert low_2se != high_2se
        assert delta["quadrature_2se"] == pytest.approx(math.sqrt(low_2se**2 + high_2se**2))
        assert delta["delta"] == pytest.approx((2 / 3) - 0.5)

    def test_a_scope_without_exactly_two_steps_is_named_in_a_warning_not_dropped_silently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        step0, step70 = self.records()
        step35 = [make_record(step=35, prompt_id="p0", sample_index=0)]
        one_step_scope = [
            make_record(
                step=0,
                section="game-behavior",
                game_id="pd-reskin",
                prompt_id="pd-reskin--other-frame--temptation-2--coop0",
                counterpart_framing=None,
            )
        ]
        records = [*step0, *step70, *step35, *one_step_scope]
        with caplog.at_level("WARNING", logger="games.trace_judge_rates"):
            rates = trace_rates({r.key: judged_row(r) for r in records}, census_rows(records))
        assert rates["deltas"] == {}
        assert f"arm-x|cell-y|{SCOPE}" in caplog.text
        assert "[0, 35, 70]" in caplog.text
        assert "arm-x|cell-y|game-behavior::pd-reskin::no-framing::temptation-2" in caplog.text
        assert "[0]" in caplog.text

    def test_drawn_subset_is_flagged_and_reported_beside_the_census(self, tmp_path: Path) -> None:
        """The overlay matches on (prompt_id, sample_index, sha256 of the completion), the handle every
        hand-read sample file carries whatever it calls its other fields."""
        step0, _ = self.records()
        judged = {r.key: judged_row(r) for r in step0}
        drawn_rows = [
            {
                "step": 0,
                "prompt_id": r.prompt_id,
                "sample_index": r.sample_index,
                "completion": r.completion,
            }
            for r in step0[:2]
        ]
        drawn_rows.append(
            {"step": 0, "prompt_id": "never-judged", "sample_index": 0, "completion": "x"}
        )
        drawn = load_sample_handles([write_jsonl(tmp_path / "drawn.jsonl", drawn_rows)])
        assert len(drawn) == 3
        rates = trace_rates(judged, census_rows(step0), drawn=drawn)
        cell = rates["cells"][CELL_0]
        assert cell["judged"] == 6, "the census reading is untouched by the overlay"
        assert cell["drawn"]["judged"] == 2
        assert cell["drawn"]["coupling_story"]["n"] == 2
        assert rates["drawn_rows_unmatched"] == 1

    def test_the_drawn_overlay_matches_on_step_so_identical_traces_at_two_steps_stay_apart(
        self, tmp_path: Path
    ) -> None:
        """A step-0 and a step-70 record of one prompt can carry byte-identical completions (a short
        deterministic trace); a handle without the step would overlay both cells from one drawn row."""
        at_0 = make_record(step=0, prompt_id="p0", sample_index=0)
        at_70 = make_record(step=70, prompt_id="p0", sample_index=0)
        assert at_0.completion_sha256 == at_70.completion_sha256
        judged = {r.key: judged_row(r) for r in (at_0, at_70)}
        drawn_row = {
            "step": 0,
            "prompt_id": "p0",
            "sample_index": 0,
            "completion": at_0.completion,
        }
        drawn = load_sample_handles([write_jsonl(tmp_path / "drawn.jsonl", [drawn_row])])
        rates = trace_rates(judged, census_rows([at_0, at_70]), drawn=drawn)
        assert rates["cells"][CELL_0]["drawn"]["judged"] == 1
        assert rates["cells"][CELL_70]["drawn"]["judged"] == 0
        assert rates["drawn_rows_unmatched"] == 0

    def test_rejects_judged_keys_missing_from_the_census(self) -> None:
        stray = make_record(prompt_id="stray")
        with pytest.raises(ValueError, match="not in the census"):
            trace_rates({stray.key: judged_row(stray)}, census_rows([make_record()]))

    def test_a_cooperators_only_census_renders_byte_identically_to_the_pre_selector_output(
        self,
    ) -> None:
        """The golden was written by the rates code before ``--decision`` and ``--per-render`` existed.
        A cooperators-only census must still render it byte for byte, whether its rows carry the
        ``decision`` stamp (written from 2026-09-03 on) or predate it (every wave-3 census)."""
        judged, census, drawn = cooperators_only_fixture()
        golden = GOLDEN_RATES.read_text(encoding="utf-8")
        assert json.dumps(trace_rates(judged, census, drawn=drawn), indent=2) + "\n" == golden
        unstamped_census = [{k: v for k, v in row.items() if k != "decision"} for row in census]
        unstamped_judged = {
            key: {k: v for k, v in row.items() if k != "decision"} for key, row in judged.items()
        }
        rendered = trace_rates(unstamped_judged, unstamped_census, drawn=drawn)
        assert json.dumps(rendered, indent=2) + "\n" == golden


def hand_label(**overrides: Any) -> dict[str, Any]:
    label: dict[str, Any] = {
        "decision_basis": "mirror",
        "counterpart_assumption": "copies-me",
        "diagonal_comparison": "only",
        "dominance_raised": True,
        "dominance_rejected": True,
        "payoff_reasoning_present": True,
        "evidence": "quoted",
        "note": "synthetic",
        "v1_decision_basis": "label-default-or-procedure",
    }
    label.update(overrides)
    return label


class TestHandLabelValidation:
    def test_full_agreement_is_green_on_every_dimension(self) -> None:
        records = [make_record(sample_index=i) for i in range(3)]
        judged = {r.key: judged_row(r) for r in records}
        report = validation_report(judged, {r.key: hand_label() for r in records})
        for dim in (
            "decision_basis",
            "counterpart_assumption",
            "diagonal_comparison",
            "dominance_raised",
            "dominance_rejected",
            "payoff_reasoning_present",
            "coupling_story",
        ):
            entry = report["dimensions"][dim]
            assert (entry["agree"], entry["lenient_agree"], entry["n"]) == (3, 3, 3), dim
            assert entry["misses"] == []
        assert report["exact_agreement"] == {"agree": 3, "n": 3, "rate": 1.0}
        assert report["dimensions"]["dominance_raised"]["caveat"] == DOMINANCE_CAVEAT
        assert "caveat" not in report["dimensions"]["decision_basis"]

    def test_sabotage_a_corrupted_expectation_goes_red(self) -> None:
        """SABOTAGE: one hand label flipped against the judged verdict must surface as a miss on its
        dimension AND drop exact agreement. A checker that stayed green here would be a message."""
        records = [make_record(sample_index=i) for i in range(3)]
        judged = {r.key: judged_row(r) for r in records}
        labels = {r.key: hand_label() for r in records}
        labels[records[1].key]["decision_basis"] = "dominance"
        report = validation_report(judged, labels)
        basis = report["dimensions"]["decision_basis"]
        assert basis["agree"] == 2
        assert basis["lenient_agree"] == 2, (
            "no secondary on either side, so lenient cannot rescue it"
        )
        assert basis["misses"] == [records[1].key]
        assert basis["confusion"]["hand=dominance|judge=mirror"] == 1
        assert report["exact_agreement"]["agree"] == 2
        assert report["dimensions"]["counterpart_assumption"]["agree"] == 3, (
            "other dimensions untouched"
        )

    def test_lenient_credits_secondary_matches_without_merging_into_strict(self) -> None:
        via_hand_secondary = make_record(sample_index=0)
        via_judge_secondary = make_record(sample_index=1)
        judged = {
            via_hand_secondary.key: judged_row(
                via_hand_secondary, decision_basis="benchmark-known-answer"
            ),
            via_judge_secondary.key: judged_row(
                via_judge_secondary,
                decision_basis="story-or-label-semantics",
                secondary_basis="mirror",
            ),
        }
        labels = {
            via_hand_secondary.key: hand_label(
                decision_basis="mirror", secondary_basis="benchmark-known-answer"
            ),
            via_judge_secondary.key: hand_label(decision_basis="mirror"),
        }
        basis = validation_report(judged, labels)["dimensions"]["decision_basis"]
        assert basis["agree"] == 0
        assert basis["lenient_agree"] == 2
        assert basis["rate"] == 0.0
        assert basis["lenient_rate"] == 1.0

    def test_levels_without_hand_positives_are_named_not_printed_as_zero_over_zero(self) -> None:
        record = make_record()
        report = validation_report({record.key: judged_row(record)}, {record.key: hand_label()})
        assumption = report["dimensions"]["counterpart_assumption"]
        assert "disclosed-and-exploited" in assumption["levels_without_hand_positives"]
        assert report["dimensions"]["diagonal_comparison"]["levels_without_hand_positives"] == [
            "none",
            "supporting",
        ]
        assert "levels_without_hand_positives" not in report["dimensions"]["dominance_raised"]

    def test_partial_labels_score_only_what_was_labelled(self) -> None:
        record = make_record()
        judged = {record.key: judged_row(record)}
        report = validation_report(judged, {record.key: {"diagonal_comparison": "none"}})
        assert set(report["dimensions"]) == {"diagonal_comparison"}
        assert report["dimensions"]["diagonal_comparison"]["agree"] == 0
        assert report["exact_agreement"] == {"agree": 0, "n": 1, "rate": 0.0}

    def test_a_label_for_an_unjudged_key_raises(self) -> None:
        record = make_record()
        with pytest.raises(ValueError, match="no judged verdict"):
            validation_report({}, {record.key: {"decision_basis": "mirror"}})

    def test_a_hand_label_that_scores_no_dimension_raises_rather_than_agreeing(self) -> None:
        """An empty or metadata-only label has nothing to disagree with, and a checker that counted it
        as exact agreement would report a calibration it never performed."""
        record = make_record()
        judged = {record.key: judged_row(record)}
        with pytest.raises(ValueError, match="scores no dimension"):
            validation_report(judged, {record.key: {}})
        with pytest.raises(ValueError, match="scores no dimension"):
            validation_report(judged, {record.key: {"note": "read it, forgot to label it"}})

    def test_an_off_enum_or_unknown_hand_field_raises_rather_than_scoring(self) -> None:
        record = make_record()
        judged = {record.key: judged_row(record)}
        with pytest.raises(ValueError, match="decision_basis"):
            validation_report(
                judged, {record.key: {"decision_basis": "label-default-or-procedure"}}
            )
        with pytest.raises(ValueError, match="unknown hand-label dimension"):
            validation_report(judged, {record.key: {"decision_base": "mirror"}})

    def test_loader_accepts_the_labels_envelope_and_the_bare_mapping(self, tmp_path: Path) -> None:
        mapping = {"k1": {"decision_basis": "mirror", "note": "n"}}
        enveloped = tmp_path / "enveloped.json"
        enveloped.write_text(
            json.dumps({"rubric_version": "x", "rubric_notes": {}, "labels": mapping})
        )
        bare = tmp_path / "bare.json"
        bare.write_text(json.dumps({"comment": "hand labels", **mapping}))
        assert load_hand_labels(enveloped) == mapping
        assert load_hand_labels(bare) == mapping

    def test_hand_keys_follow_the_exports_spelling_and_resolve_through_the_digest(
        self, tmp_path: Path
    ) -> None:
        """The ladder export keys by rung + label_print_order, the control export by source_arm + order,
        and neither spelling is the judged rows' arm|cell; the completion digest is the shared identity."""
        ladder_record = make_record(step=0, sample_index=4)
        control_record = make_record(
            step=70, sample_index=5, completion="another trace</think>" + VISIBLE
        )
        judged = {r.key: judged_row(r) for r in (ladder_record, control_record)}
        ladder_row = {
            "rung": "other",
            "recipient": "theirs",
            "cell": "twin-pd::unstated::temptation-2@0",
            "step": 0,
            "prompt_id": ladder_record.prompt_id,
            "sample_index": 4,
            "payoff_variant": "temptation-2",
            "label_print_order": "canonical",
            "completion": ladder_record.completion,
            "visible_text": VISIBLE,
        }
        control_row = {
            "cell": "held-out-skins@70",
            "step": 70,
            "source_arm": "pd-reskin-msfp",
            "prompt_id": control_record.prompt_id,
            "order": "canonical",
            "sample_index": 5,
            "completion": control_record.completion,
            "visible_text": VISIBLE,
        }
        ladder_key = sample_row_key(ladder_row)
        control_key = sample_row_key(control_row)
        assert (
            ladder_key
            == f"other|twin-pd::unstated::temptation-2@0|0|{ladder_record.prompt_id}|4|canonical"
        )
        assert (
            control_key
            == f"pd-reskin-msfp|held-out-skins@70|70|{control_record.prompt_id}|5|canonical"
        )
        ladder_path = write_jsonl(tmp_path / "ladder.jsonl", [ladder_row])
        control_path = write_jsonl(tmp_path / "control.jsonl", [control_row])
        labels = {
            ladder_key: hand_label(source_file=str(ladder_path.relative_to(tmp_path))),
            control_key: hand_label(source_file=str(control_path.relative_to(tmp_path))),
        }
        assert hand_label_sample_files(labels, relative_to=tmp_path) == [control_path, ladder_path]
        key_map = resolve_hand_keys(labels, judged, [ladder_path, control_path])
        assert key_map == {ladder_key: ladder_record.key, control_key: control_record.key}
        report = validation_report(judged, labels, key_map=key_map)
        assert report["exact_agreement"]["agree"] == 2
        with pytest.raises(ValueError, match="no judged verdict"):
            resolve_hand_keys(
                {**labels, "self|missing@0|0|x|0|canonical": hand_label()}, judged, [ladder_path]
            )
        # An unrelated completion in the sample file cannot be mistaken for the judged record.
        stale = dict(ladder_row, completion="not what was judged")
        with pytest.raises(ValueError, match="no judged verdict"):
            resolve_hand_keys(
                {ladder_key: hand_label()}, judged, [write_jsonl(tmp_path / "stale.jsonl", [stale])]
            )

    def test_hand_keys_for_identical_traces_at_two_steps_resolve_to_their_own_rows(
        self, tmp_path: Path
    ) -> None:
        at_0 = make_record(step=0, prompt_id="p0", sample_index=0)
        at_70 = make_record(step=70, prompt_id="p0", sample_index=0)
        judged = {r.key: judged_row(r) for r in (at_0, at_70)}
        rows = [
            {
                "rung": "other",
                "cell": f"twin-pd::unstated::temptation-2@{record.step}",
                "step": record.step,
                "prompt_id": "p0",
                "sample_index": 0,
                "label_print_order": "canonical",
                "completion": record.completion,
            }
            for record in (at_0, at_70)
        ]
        path = write_jsonl(tmp_path / "ladder.jsonl", rows)
        keys = [sample_row_key(row) for row in rows]
        key_map = resolve_hand_keys({k: hand_label() for k in keys}, judged, [path])
        assert key_map == {keys[0]: at_0.key, keys[1]: at_70.key}

    def test_the_record_handle_is_the_judged_rows_digest(self) -> None:
        record = make_record()
        assert record.completion_sha256 == hashlib.sha256(COMPLETION.encode()).hexdigest()
        assert record.handle == (
            record.step,
            record.prompt_id,
            record.sample_index,
            record.completion_sha256,
        )
