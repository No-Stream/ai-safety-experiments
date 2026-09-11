"""The trace judge's record selectors: ``--decision`` and ``--per-render`` on the loader, the decision
stamp on census and judged rows, the manifest both commands write, and the rates header and marker.

Fixtures are shared with ``test_games_trace_judge`` (synthetic traces only; this file is tracked). The
selectors' semantics are the 2026-09-03 scratch driver's, which the 9B coherence census validated;
here they are pinned so the next wave never needs a driver.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import TYPE_CHECKING, Any

import pytest

import games.trace_judge as trace_judge_module
from games.prompts import LABEL_PRINT_ORDER_SWAPPED
from games.tests.test_games_trace_judge import (
    VERDICT_PAYLOAD,
    ScriptedBackend,
    battery_row,
    run_judge,
    write_cell,
)
from games.trace_judge import declared_per_render, load_census, load_judged, main, write_census
from games.trace_judge_rates import DECISION_COOPERATE, DECISION_DEFECT
from games.trace_judge_records import (
    DECISION_ALL,
    DECISION_CHOICES,
    Selection,
    load_cell_records,
    record_decision,
)

if TYPE_CHECKING:
    from pathlib import Path

    from games.trace_judge_records import TraceRecord

ARM = "arm-dir"
CELL = "cell-y"
STEP = 70
CANONICAL = "canonical"
ORDERS = (CANONICAL, LABEL_PRINT_ORDER_SWAPPED)
UNSTATED_PROMPT = "twin-pd--frame--temptation-2--framing-unstated--coop0"
HUMAN_PROMPT = "twin-pd--frame--temptation-2--framing-human--coop0"


def defector_row(**overrides: Any) -> dict[str, Any]:
    return battery_row(action="D", coop_fraction=0.0, **overrides)


def unparsed_row(**overrides: Any) -> dict[str, Any]:
    return battery_row(action=None, coop_fraction=None, parsed=False, **overrides)


def human_defector_row(**overrides: Any) -> dict[str, Any]:
    return defector_row(prompt_id=HUMAN_PROMPT, counterpart_framing="human", **overrides)


def mixed_cell_rows() -> list[dict[str, Any]]:
    """Both renders of one prompt: cooperators at the even sample indices, defectors at the odd ones and
    one unparsed row each; plus a second prompt (the human framing) with a single defector."""
    rows: list[dict[str, Any]] = []
    for order in ORDERS:
        for index in range(6):
            maker = battery_row if index % 2 == 0 else defector_row
            rows.append(maker(sample_index=index, label_print_order=order))
        rows.append(unparsed_row(sample_index=6, label_print_order=order))
    rows.append(human_defector_row(sample_index=0))
    return rows


MIXED_FUNNEL = {
    "rows": 15,
    "matrix_game_rows": 15,
    "in_scope_rows": 15,
    "parsed": 13,
    "cooperating": 6,
    "defecting": 7,
}


def load_rows(
    tree: Path, rows: list[dict[str, Any]], selection: Selection
) -> tuple[list[TraceRecord], dict[str, int]]:
    write_cell(tree, ARM, CELL, STEP, rows)
    return load_cell_records(tree, ARM, CELL, STEP, selection)


def key_of(prompt_id: str, sample_index: int, order: str = CANONICAL) -> str:
    return f"{ARM}|{CELL}|{STEP}|{prompt_id}|{sample_index}|{order}"


def cell_args(tree: Path, out: Path) -> list[str]:
    return [
        "--tree",
        str(tree),
        "--arm",
        ARM,
        "--cell",
        CELL,
        "--steps",
        str(STEP),
        "--out",
        str(out),
    ]


def the_one_manifest(out: Path) -> dict[str, Any]:
    manifests = sorted(out.glob("manifest-*.json"))
    assert len(manifests) == 1, [p.name for p in manifests]
    return json.loads(manifests[0].read_text(encoding="utf-8"))


class TestDecisionSelection:
    def test_the_default_is_cooperators_only_and_equals_an_explicit_cooperate(
        self, tmp_path: Path
    ) -> None:
        """Today's behaviour, byte for byte: the same keys in file order, every one a cooperator."""
        assert Selection().decision == DECISION_COOPERATE
        assert Selection().per_render is None
        default, default_counts = load_rows(tmp_path / "default", mixed_cell_rows(), Selection())
        explicit, explicit_counts = load_rows(
            tmp_path / "explicit", mixed_cell_rows(), Selection(decision=DECISION_COOPERATE)
        )
        assert [r.key for r in default] == [r.key for r in explicit]
        assert [r.key for r in default] == [
            key_of(UNSTATED_PROMPT, index, order) for order in ORDERS for index in (0, 2, 4)
        ]
        assert {r.decision for r in default} == {DECISION_COOPERATE}
        assert default_counts == explicit_counts
        assert default_counts == {**MIXED_FUNNEL, "selected": 6, "after_per_render": 6}

    def test_defect_selects_parsed_full_defectors_only(self, tmp_path: Path) -> None:
        """SABOTAGE target: a rule that read 1.0 as the defector value selects the cooperators here."""
        records, counts = load_rows(
            tmp_path, mixed_cell_rows(), Selection(decision=DECISION_DEFECT)
        )
        assert [r.key for r in records] == [
            *(key_of(UNSTATED_PROMPT, index, order) for order in ORDERS for index in (1, 3, 5)),
            key_of(HUMAN_PROMPT, 0),
        ]
        assert {r.decision for r in records} == {DECISION_DEFECT}
        assert counts == {**MIXED_FUNNEL, "selected": 7, "after_per_render": 7}
        cooperators, _ = load_rows(tmp_path / "coop", mixed_cell_rows(), Selection())
        assert not {r.key for r in cooperators} & {r.key for r in records}

    def test_all_selects_every_parsed_record_stamped_with_its_own_decision(
        self, tmp_path: Path
    ) -> None:
        """The stamp is the record's, never the selector's, and cooperators and defectors of one render
        sit under different keys because the key carries the sample index."""
        records, counts = load_rows(tmp_path, mixed_cell_rows(), Selection(decision=DECISION_ALL))
        assert counts == {**MIXED_FUNNEL, "selected": 13, "after_per_render": 13}
        keys = [r.key for r in records]
        assert len(set(keys)) == len(keys) == 13
        assert Counter(r.decision for r in records) == {DECISION_COOPERATE: 6, DECISION_DEFECT: 7}
        for record in records:
            expected = (
                DECISION_DEFECT
                if record.prompt_id == HUMAN_PROMPT or record.sample_index % 2
                else DECISION_COOPERATE
            )
            assert record.decision == expected, record.key
        cooperators, _ = load_rows(tmp_path / "coop", mixed_cell_rows(), Selection())
        defectors, _ = load_rows(
            tmp_path / "defect", mixed_cell_rows(), Selection(decision=DECISION_DEFECT)
        )
        assert set(keys) == {r.key for r in cooperators} | {r.key for r in defectors}

    def test_a_fractional_coop_fraction_on_a_matrix_game_row_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A one-shot matrix action scores exactly 1.0 or 0.0; anything else is an upstream scoring change
        and must not be filed under a decision the record never took."""
        assert record_decision({"coop_fraction": 1.0}) == DECISION_COOPERATE
        assert record_decision({"coop_fraction": 0.0}) == DECISION_DEFECT
        with pytest.raises(ValueError, match=r"neither 1\.0 nor 0\.0"):
            load_rows(tmp_path, [battery_row(coop_fraction=0.5)], Selection(decision=DECISION_ALL))

    def test_an_unknown_decision_or_a_non_positive_per_render_is_refused(self) -> None:
        assert DECISION_CHOICES == ("cooperate", "defect", "all")
        with pytest.raises(ValueError, match="decision must be one of"):
            Selection(decision="coop")
        with pytest.raises(ValueError, match="per_render must be at least 1"):
            Selection(per_render=0)

    def test_the_census_and_the_judged_rows_carry_the_decision(self, tmp_path: Path) -> None:
        records, _ = load_rows(tmp_path, mixed_cell_rows(), Selection(decision=DECISION_DEFECT))
        out = tmp_path / "out"
        out.mkdir()
        write_census(out / "census.jsonl", records)
        assert {row["decision"] for row in load_census(out / "census.jsonl")} == {DECISION_DEFECT}
        backend = ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)} for _ in records])
        run_judge(backend, records, out / "judged.jsonl")
        judged = load_judged(out / "judged.jsonl")
        assert len(judged) == 7
        assert {row["decision"] for row in judged.values()} == {DECISION_DEFECT}


class TestPerRenderSampling:
    def test_keeps_the_lowest_sample_indices_numerically_per_prompt_render(
        self, tmp_path: Path
    ) -> None:
        """Indices run past 9 so a lexical sort (0, 1, 10) would fail where the numeric one (0, 1, 2)
        passes; each print order is its own render and keeps its own lowest three."""
        rows = [
            battery_row(sample_index=index, label_print_order=order)
            for order in ORDERS
            for index in range(12)
        ]
        records, counts = load_rows(tmp_path, rows, Selection(per_render=3))
        assert [(r.label_print_order, r.sample_index) for r in records] == [
            (CANONICAL, 0),
            (CANONICAL, 1),
            (CANONICAL, 2),
            (LABEL_PRINT_ORDER_SWAPPED, 0),
            (LABEL_PRINT_ORDER_SWAPPED, 1),
            (LABEL_PRINT_ORDER_SWAPPED, 2),
        ]
        assert counts["selected"] == 24
        assert counts["after_per_render"] == 6

    def test_is_independent_of_file_order(self, tmp_path: Path) -> None:
        """Shuffle the cell file: the same keys come out in the same order, because the sample derives
        from record identity (prompt, print order, sample index) and never from where a row sat."""
        rows = [
            battery_row(sample_index=index, label_print_order=order)
            for order in ORDERS
            for index in range(8)
        ] + [human_defector_row(sample_index=index) for index in range(4)]
        shuffled = list(rows)
        random.Random(7).shuffle(shuffled)
        assert shuffled != rows
        selection = Selection(decision=DECISION_ALL, per_render=2)
        in_order, _ = load_rows(tmp_path / "in-order", rows, selection)
        from_shuffled, _ = load_rows(tmp_path / "shuffled", shuffled, selection)
        assert [r.key for r in in_order] == [r.key for r in from_shuffled]
        assert [r.key for r in in_order] == [
            key_of(HUMAN_PROMPT, 0),
            key_of(HUMAN_PROMPT, 1),
            key_of(UNSTATED_PROMPT, 0),
            key_of(UNSTATED_PROMPT, 1),
            key_of(UNSTATED_PROMPT, 0, LABEL_PRINT_ORDER_SWAPPED),
            key_of(UNSTATED_PROMPT, 1, LABEL_PRINT_ORDER_SWAPPED),
        ]
        assert selection.sample_per_render(list(reversed(in_order))) == in_order

    def test_under_all_keeps_the_lowest_indices_whatever_decision_they_took(
        self, tmp_path: Path
    ) -> None:
        records, counts = load_rows(
            tmp_path, mixed_cell_rows(), Selection(decision=DECISION_ALL, per_render=2)
        )
        assert [
            (r.prompt_id, r.label_print_order, r.sample_index, r.decision) for r in records
        ] == [
            (HUMAN_PROMPT, CANONICAL, 0, DECISION_DEFECT),
            (UNSTATED_PROMPT, CANONICAL, 0, DECISION_COOPERATE),
            (UNSTATED_PROMPT, CANONICAL, 1, DECISION_DEFECT),
            (UNSTATED_PROMPT, LABEL_PRINT_ORDER_SWAPPED, 0, DECISION_COOPERATE),
            (UNSTATED_PROMPT, LABEL_PRINT_ORDER_SWAPPED, 1, DECISION_DEFECT),
        ]
        assert counts == {**MIXED_FUNNEL, "selected": 13, "after_per_render": 5}

    def test_the_selection_manifest_carries_both_selectors(self) -> None:
        assert Selection(decision=DECISION_DEFECT, per_render=1).as_manifest() == {
            "sections": None,
            "games": None,
            "framings": None,
            "decision": DECISION_DEFECT,
            "per_render": 1,
        }
        default = Selection().as_manifest()
        assert default["decision"] == DECISION_COOPERATE
        assert default["per_render"] is None


class TestSelectorCLI:
    def test_keyword_stamps_the_selection_into_its_manifest_and_census(
        self, tmp_path: Path
    ) -> None:
        """The companion takes the same selectors and writes the same census and manifest, so a keyword-only
        directory says what it holds too; a per-render sample here is one record per render."""
        write_cell(tmp_path, ARM, CELL, STEP, mixed_cell_rows())
        out = tmp_path / "out"
        main(["keyword", *cell_args(tmp_path, out), "--decision", "defect", "--per-render", "1"])
        manifest = the_one_manifest(out)
        assert manifest["command"] == "keyword"
        assert manifest["selection"] == {
            "sections": None,
            "games": None,
            "framings": None,
            "decision": "defect",
            "per_render": 1,
        }
        assert manifest["funnel"][str(STEP)] == {
            **MIXED_FUNNEL,
            "selected": 7,
            "after_per_render": 3,
        }
        assert manifest["counts"] == {"records": 3, "keyword_rows": 3}
        census = load_census(out / "census.jsonl")
        assert sorted(row["key"] for row in census) == sorted(
            [
                key_of(HUMAN_PROMPT, 0),
                key_of(UNSTATED_PROMPT, 1),
                key_of(UNSTATED_PROMPT, 1, LABEL_PRINT_ORDER_SWAPPED),
            ]
        )
        assert {row["decision"] for row in census} == {"defect"}
        keyword_rows = [
            json.loads(line) for line in (out / "keyword.jsonl").read_text().splitlines()
        ]
        assert {row["decision"] for row in keyword_rows} == {"defect"}
        assert declared_per_render(out) == (1, True)

    def test_judge_stamps_the_selection_and_judges_only_the_selected_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_cell(tmp_path, ARM, CELL, STEP, mixed_cell_rows())
        backend = ScriptedBackend([{"text": json.dumps(VERDICT_PAYLOAD)} for _ in range(7)])
        monkeypatch.setattr(trace_judge_module, "BedrockBackend", lambda *_a, **_k: backend)
        out = tmp_path / "out"
        main(["judge", *cell_args(tmp_path, out), "--decision", "defect"])
        assert len(backend.prompts_seen) == 7
        judged = load_judged(out / "judged.jsonl")
        assert len(judged) == 7
        assert {row["decision"] for row in judged.values()} == {"defect"}
        manifest = the_one_manifest(out)
        assert manifest["command"] == "judge"
        assert manifest["selection"]["decision"] == "defect"
        assert manifest["selection"]["per_render"] is None
        assert manifest["counts"]["attempted"] == 7
        assert manifest["funnel"][str(STEP)]["selected"] == 7
        assert set(manifest["usage"]) == {
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
        }
        assert declared_per_render(out) == (None, True)

    def test_rates_marks_a_sampled_directory_and_says_so_in_its_header(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A sample's rates file carries the marker first; a census run's file is unchanged and the header
        says census. Under ``all`` the defector cells carry their own scope label beside the bare one."""
        write_cell(tmp_path, ARM, CELL, STEP, mixed_cell_rows())
        sample_dir = tmp_path / "sample"
        main(
            ["keyword", *cell_args(tmp_path, sample_dir), "--decision", "all", "--per-render", "1"]
        )
        (sample_dir / "judged.jsonl").write_text("")
        with caplog.at_level("INFO", logger="games.trace_judge"):
            main(
                [
                    "rates",
                    "--judged",
                    str(sample_dir / "judged.jsonl"),
                    "--out",
                    str(sample_dir / "rates.json"),
                ]
            )
        rates = json.loads((sample_dir / "rates.json").read_text())
        assert next(iter(rates)) == "selection"
        assert rates["selection"]["per_render"] == 1
        assert "SAMPLE, not a census" in rates["selection"]["sampling"]
        assert "selection: per-render 1" in caplog.text
        assert "decisions in the census: cooperate, defect" in caplog.text
        # Index 0 of both unstated renders is a cooperator; the human render's is a defector.
        assert set(rates["cells"]) == {
            f"{ARM}|{CELL}|framing-sweep::twin-pd::unstated::temptation-2@{STEP}",
            f"{ARM}|{CELL}|framing-sweep::twin-pd::human::temptation-2::defect@{STEP}",
        }
        assert (
            rates["cells"][f"{ARM}|{CELL}|framing-sweep::twin-pd::unstated::temptation-2@{STEP}"][
                "examined"
            ]
            == 2
        )
        caplog.clear()
        census_dir = tmp_path / "census"
        main(["keyword", *cell_args(tmp_path, census_dir)])
        (census_dir / "judged.jsonl").write_text("")
        with caplog.at_level("INFO", logger="games.trace_judge"):
            main(
                [
                    "rates",
                    "--judged",
                    str(census_dir / "judged.jsonl"),
                    "--out",
                    str(census_dir / "rates.json"),
                ]
            )
        census_rates = json.loads((census_dir / "rates.json").read_text())
        assert "selection" not in census_rates
        assert list(census_rates) == ["cells", "prompts", "deltas", "dimensions"]
        assert "selection: census" in caplog.text

    def test_rates_refuses_a_directory_that_mixes_a_sample_with_a_population(
        self, tmp_path: Path
    ) -> None:
        """The census is the rates' examined denominator; one directory holding both would pool a sample
        with a population under one scope label. Two launches in one second must not collide either."""
        write_cell(tmp_path, ARM, CELL, STEP, mixed_cell_rows())
        out = tmp_path / "out"
        main(["keyword", *cell_args(tmp_path, out), "--per-render", "1"])
        main(["keyword", *cell_args(tmp_path, out)])
        assert len(list(out.glob("manifest-*.json"))) == 2
        with pytest.raises(ValueError, match="different per-render samplings"):
            main(["rates", "--judged", str(out / "judged.jsonl"), "--out", str(out / "rates.json")])
        assert not (out / "rates.json").exists()

    def test_declared_per_render_reads_old_and_scratch_driver_manifests(
        self, tmp_path: Path
    ) -> None:
        """A pre-selector manifest declares nothing and is the whole population; the 2026-09-03 scratch
        driver carried ``per_render`` at the top level, and its sample directories must still read as
        samples; a directory with no manifest says so rather than assuming a census."""
        old = tmp_path / "old"
        old.mkdir()
        (old / "manifest-20260902T000000Z.json").write_text(
            json.dumps({"selection": {"sections": None, "games": None, "framings": None}})
        )
        assert declared_per_render(old) == (None, True)
        driver = tmp_path / "driver"
        driver.mkdir()
        (driver / "manifest-20260903T000000Z.json").write_text(
            json.dumps(
                {
                    "driver": "judge_select.py",
                    "selection": {"sections": None, "games": None, "framings": None},
                    "decision": "defect",
                    "per_render": 1,
                }
            )
        )
        assert declared_per_render(driver) == (1, True)
        empty = tmp_path / "empty"
        empty.mkdir()
        assert declared_per_render(empty) == (None, False)
