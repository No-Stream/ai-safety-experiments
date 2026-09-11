"""Offline tests for re-emitting a selected corpus under a different grading.

The transform exists to keep the twin-pd-group / twin-pd-self contrast honest: one sweep, one
selected prompt set, two gradings. So the tests care most about the guard that would stop it being
used where it is unsound.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games import regrade_corpus as rc

if TYPE_CHECKING:
    from pathlib import Path


def corpus_rows(
    n: int = 3, *, grading: str = "group-mix", game_id: str = "twin-pd"
) -> list[dict[str, Any]]:
    """Rows shaped like what games.select_prompts writes, including a sweep-added column."""
    return [
        {
            "prompt_id": f"twin-pd--frame-{index}",
            "raw_prompt": "a shift sheet",
            "prompt": "<|im_start|>user\na shift sheet<|im_end|>\n",
            "game_id": game_id,
            "grading": grading,
            "payoff_variant": "temptation-2",
            "label_a": "SIDE",
            "label_b": "LOOP",
            "coop_label": "SIDE",
            "coop_rate_at_selection": 0.5,
        }
        for index in range(n)
    ]


class TestRegradeRows:
    def test_only_the_grading_column_changes(self):
        before = corpus_rows()
        after = rc.regrade_rows(before, target_grading="self")
        assert [row["grading"] for row in after] == ["self"] * len(before)
        for original, regraded in zip(before, after, strict=True):
            assert set(original) == set(regraded)
            differing = {key for key in original if original[key] != regraded[key]}
            assert differing == {"grading"}

    def test_sweep_added_columns_survive(self):
        # The re-graded corpus must keep its provenance: it is the same selected prompt set.
        after = rc.regrade_rows(corpus_rows(), target_grading="self")
        assert all(row["coop_rate_at_selection"] == 0.5 for row in after)

    def test_the_input_rows_are_not_mutated(self):
        before = corpus_rows()
        rc.regrade_rows(before, target_grading="self")
        assert {row["grading"] for row in before} == {"group-mix"}

    def test_a_no_op_regrade_is_refused(self):
        # Writing a copy that is already graded the same way would later read as a second,
        # independent selection, which is precisely the confusion this transform avoids.
        with pytest.raises(ValueError, match="already graded"):
            rc.regrade_rows(corpus_rows(), target_grading="group-mix")

    def test_a_mixed_grading_corpus_is_refused(self):
        rows = corpus_rows(2) + corpus_rows(1, grading="self")
        with pytest.raises(ValueError, match="mixes gradings"):
            rc.regrade_rows(rows, target_grading="self")

    def test_a_mixed_game_corpus_regrades_every_row(self):
        # Refused until wave 4b, and now the case the breadth arms need: one corpus carrying five
        # matrix games plus the trust sender under a single grading, whose alpha-0 control arm is that
        # same file regraded. Nothing about the transform was game-specific; the refusal was.
        rows = corpus_rows(2) + corpus_rows(1, game_id="stag-hunt")
        after = rc.regrade_rows(rows, target_grading="self")
        assert [row["grading"] for row in after] == ["self"] * 3
        assert [row["game_id"] for row in after] == ["twin-pd", "twin-pd", "stag-hunt"]

    def test_every_distinct_game_in_a_mixed_corpus_is_checked_for_independence(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # The property the transform rests on is a property of each GAME's renderer, so a loop that
        # checked only the first row's game would pass a corpus whose other games condition their
        # text on the grading -- the exact silent case the guard exists for.
        checked: list[str] = []
        monkeypatch.setattr(
            rc,
            "assert_grading_is_prompt_independent",
            lambda game_id, source, target: checked.append(f"{game_id}:{source}->{target}"),
        )
        rows = (
            corpus_rows(2) + corpus_rows(1, game_id="stag-hunt") + corpus_rows(1, game_id="chicken")
        )
        rc.regrade_rows(rows, target_grading="self")
        assert checked == [
            "chicken:group-mix->self",
            "stag-hunt:group-mix->self",
            "twin-pd:group-mix->self",
        ]

    def test_a_mixed_game_corpus_regrades_between_two_care_weights(self):
        # The wave-4b control's own transform, against the live generator for each game.
        rows = corpus_rows(2, grading="care-alpha-1") + corpus_rows(
            1, grading="care-alpha-1", game_id="stag-hunt"
        )
        after = rc.regrade_rows(rows, target_grading="care-alpha-0")
        assert {row["grading"] for row in after} == {"care-alpha-0"}

    def test_a_non_canonical_care_spelling_is_refused_as_a_target(self):
        with pytest.raises(ValueError, match="use 'care-alpha-1'"):
            rc.regrade_rows(corpus_rows(), target_grading="care-alpha-1.0")

    def test_an_unknown_grading_is_refused(self):
        with pytest.raises(ValueError, match="Unknown grading"):
            rc.regrade_rows(corpus_rows(), target_grading="vibes")

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(ValueError, match="no rows"):
            rc.regrade_rows([], target_grading="self")

    def test_rows_without_the_needed_columns_are_refused(self):
        with pytest.raises(ValueError, match="carry no"):
            rc.regrade_rows([{"prompt_id": "x"}], target_grading="self")


class TestGradingIndependenceGuard:
    """The guard is what makes this transform safe for games it was never tested against."""

    def test_twin_pd_prompts_really_are_grading_independent(self):
        # Verified against the live generator rather than assumed: this is the property the whole
        # twin-pd-group / twin-pd-self contrast rests on.
        rc.assert_grading_is_prompt_independent("twin-pd", "group-mix", "self")

    @pytest.mark.parametrize(
        "ladder_grading", ["joint-welfare-group-mix", "other-payoff-group-mix"]
    )
    def test_the_grading_ladder_regrades_the_unstated_corpus(self, ladder_grading: str):
        # The wave-3 ladder arms inherit the pd-unstated selection by regrading its banked
        # group-mix corpus, so this seam -- not a fresh sweep -- is what their trainability
        # rests on, exactly as the format-only placebo's did.
        rc.assert_grading_is_prompt_independent("pd-unstated", "group-mix", ladder_grading)
        after = rc.regrade_rows(corpus_rows(game_id="pd-unstated"), target_grading=ladder_grading)
        assert {row["grading"] for row in after} == {ladder_grading}

    def test_a_game_whose_prompts_depend_on_the_grading_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # No such game exists today, so the failure mode is induced. Without this the guard would
        # be a check nobody has ever watched fail.
        def grading_dependent(game_id: str, grading: str, *, split: str) -> list[dict[str, Any]]:
            del game_id, split
            return [{"grading": grading, "raw_prompt": f"scored by {grading}"}]

        monkeypatch.setattr(rc, "generate_prompt_rows", grading_dependent)
        with pytest.raises(ValueError, match="not grading-independent"):
            rc.assert_grading_is_prompt_independent("twin-pd", "group-mix", "self")

    def test_a_game_that_renders_a_different_row_count_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def uneven(game_id: str, grading: str, *, split: str) -> list[dict[str, Any]]:
            del game_id, split
            return [{"grading": grading}] * (1 if grading == "self" else 2)

        monkeypatch.setattr(rc, "generate_prompt_rows", uneven)
        with pytest.raises(ValueError, match="not the same corpus"):
            rc.assert_grading_is_prompt_independent("twin-pd", "group-mix", "self")

    def test_a_game_that_renders_different_columns_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def uneven_columns(game_id: str, grading: str, *, split: str) -> list[dict[str, Any]]:
            del game_id, split
            row: dict[str, Any] = {"grading": grading}
            if grading == "self":
                row["extra"] = 1
            return [row]

        monkeypatch.setattr(rc, "generate_prompt_rows", uneven_columns)
        with pytest.raises(ValueError, match="different columns"):
            rc.assert_grading_is_prompt_independent("twin-pd", "group-mix", "self")


class TestCli:
    def test_the_grading_flag_accepts_a_care_family_member(self, tmp_path: Path):
        # `choices=` cannot express a family whose alpha is a number, so the flag validates through
        # games.rewards.grading_cli_value instead; without it every care regrade dies at argparse.
        source = tmp_path / "corpus-mixed.jsonl"
        rows = corpus_rows(2, grading="care-alpha-1") + corpus_rows(
            1, grading="care-alpha-1", game_id="chicken"
        )
        source.write_text("\n".join(json.dumps(row) for row in rows))
        destination = tmp_path / "corpus-mixed-self.jsonl"
        rc.main(["--corpus", str(source), "--grading", "care-alpha-0", "--out", str(destination)])
        assert {row["grading"] for row in rc.read_corpus(destination)} == {"care-alpha-0"}

    def test_the_grading_flag_refuses_a_name_no_grading_answers_to(self, tmp_path: Path):
        source = tmp_path / "corpus.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in corpus_rows(2)))
        with pytest.raises(SystemExit):
            rc.main(["--corpus", str(source), "--grading", "care-alpha-x", "--out", "unused.jsonl"])

    def test_a_corpus_round_trips_through_the_command_line(self, tmp_path: Path):
        source = tmp_path / "corpus-twin-pd.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in corpus_rows(4)))
        destination = tmp_path / "corpus-twin-pd-self.jsonl"
        rc.main(["--corpus", str(source), "--grading", "self", "--out", str(destination)])
        written = rc.read_corpus(destination)
        assert len(written) == 4
        assert {row["grading"] for row in written} == {"self"}
        assert {row["prompt_id"] for row in written} == {row["prompt_id"] for row in corpus_rows(4)}

    def test_blank_lines_are_tolerated_on_read(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        rows = corpus_rows(2)
        path.write_text(json.dumps(rows[0]) + "\n\n" + json.dumps(rows[1]) + "\n")
        assert len(rc.read_corpus(path)) == 2

    def test_the_written_file_is_headerless(self, tmp_path: Path):
        # games/dataset.py feeds these straight into Dataset.from_list, so a meta line would
        # arrive as a row with every column missing.
        path = tmp_path / "out.jsonl"
        rc.write_corpus(path, corpus_rows(2))
        first = json.loads(path.read_text().splitlines()[0])
        assert "prompt_id" in first
