"""Offline tests for `games/render_corpus.py`, the pre-selection corpus render CLI.

Everything here is CPU and deterministic: rendering prompts touches no tokenizer and no model, so
these tests exercise the real `games.prompts` renderer and the real `games.select_prompts` writer
rather than fixtures shaped like them.

Two claims carry the amplification read this CLI exists for, and both are asserted directly rather
than assumed. First, the rendered file is `generate_prompt_rows` verbatim -- so a per-skin step-0
versus step-70 comparison run over it covers the prompts the baseline selection swept out, which are
precisely the near-zero and near-one skins the read is about. Second, the file is the same artifact
schema the selection sweep writes, checked by loading it through
`games.eval_training_frames.load_corpus_rows`, the downstream consumer, and by showing that the
sweep-then-regrade path and the direct render path produce identical bytes.

Per the repo rule that a check you have never watched fail is not yet a check, the refusals are
exercised by committing the violation each exists to catch: an `--out` that already exists, and a
game the renderer cannot render.

The row counts below are restated rather than imported from `test_games_prompts`, so a renderer
that quietly stopped emitting the held-out skins fails here too instead of both pins moving
together.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games import eval_training_frames, regrade_corpus, render_corpus
from games.prompts import LABEL_PRINT_ORDER_CANONICAL, generate_prompt_rows

if TYPE_CHECKING:
    from pathlib import Path

GAME = "pd-reskin"
GRADING = "self"
PD_RESKIN_TRAIN_ROWS = 176
PD_RESKIN_EVAL_ROWS = 48

GRADING_COLUMN = "grading"


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a headerless corpus JSONL back, the way every consumer of one does."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class TestRenderCorpus:
    def test_train_split_renders_every_prompt_selection_would_narrow(self, tmp_path: Path):
        out = tmp_path / "pd-reskin-self-train.jsonl"
        rows = render_corpus.render_corpus(out, game_id=GAME, grading=GRADING, split="train")
        assert len(rows) == PD_RESKIN_TRAIN_ROWS
        assert read_rows(out) == generate_prompt_rows(GAME, GRADING, split="train")

    def test_eval_split_renders_the_held_out_skins(self, tmp_path: Path):
        out = tmp_path / "pd-reskin-self-eval.jsonl"
        rows = render_corpus.render_corpus(out, game_id=GAME, grading=GRADING, split="eval")
        assert len(rows) == PD_RESKIN_EVAL_ROWS
        assert read_rows(out) == generate_prompt_rows(GAME, GRADING, split="eval")

    def test_the_default_label_print_order_is_the_canonical_one(self, tmp_path: Path):
        implicit = tmp_path / "implicit.jsonl"
        explicit = tmp_path / "explicit.jsonl"
        render_corpus.render_corpus(implicit, game_id=GAME, grading=GRADING, split="train")
        render_corpus.render_corpus(
            explicit,
            game_id=GAME,
            grading=GRADING,
            split="train",
            label_print_order=LABEL_PRINT_ORDER_CANONICAL,
        )
        assert implicit.read_bytes() == explicit.read_bytes()

    def test_a_swapped_print_order_renders_a_different_prompt_set(self, tmp_path: Path):
        """Prove the print-order flag reaches the renderer at all.

        An argument accepted and dropped would return the canonical rows under a column claiming
        otherwise, which is how a pooled corpus stops saying what is in it.
        """
        out = tmp_path / "swapped.jsonl"
        rows = render_corpus.render_corpus(
            out, game_id=GAME, grading=GRADING, split="train", label_print_order="swapped"
        )
        assert rows == generate_prompt_rows(
            GAME, GRADING, split="train", label_print_order="swapped"
        )
        assert {str(row["label_print_order"]) for row in rows} == {"swapped"}

    def test_refuses_an_out_file_that_already_exists(self, tmp_path: Path):
        out = tmp_path / "already-there.jsonl"
        out.write_text("a corpus somebody else rendered\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="already exists"):
            render_corpus.render_corpus(out, game_id=GAME, grading=GRADING, split="train")
        assert out.read_text(encoding="utf-8") == "a corpus somebody else rendered\n"

    def test_an_unrenderable_game_raises_the_renderers_own_error(self, tmp_path: Path):
        """The CLI does not re-validate game, grading, split or print order.

        `games.prompts._assert_generatable` already names what would have been produced instead of
        refusing, so a second copy of those checks here would answer with a worse message.
        """
        out = tmp_path / "never-written.jsonl"
        with pytest.raises(ValueError, match="Unknown game_id"):
            render_corpus.render_corpus(out, game_id="not-a-game", grading=GRADING, split="train")
        assert not out.exists()

    def test_an_eval_only_game_still_refuses_a_train_split(self, tmp_path: Path):
        out = tmp_path / "never-written.jsonl"
        with pytest.raises(ValueError, match="eval-only"):
            render_corpus.render_corpus(
                out, game_id="twin-pd-temptation-dose", grading=GRADING, split="train"
            )
        assert not out.exists()


class TestRenderCorpusCli:
    def test_the_cli_writes_what_the_function_writes(self, tmp_path: Path):
        out = tmp_path / "nested" / "pd-reskin-self-train.jsonl"
        render_corpus.main(
            [
                "--game",
                GAME,
                "--grading",
                GRADING,
                "--split",
                "train",
                "--out",
                str(out),
            ]
        )
        assert read_rows(out) == generate_prompt_rows(GAME, GRADING, split="train")


class TestDownstreamConsumers:
    def test_the_rendered_file_loads_through_the_frames_evaluator(self, tmp_path: Path):
        """`games.eval_training_frames --corpus` is what the amplification read runs.

        It refuses anything that is not a selection artifact, so passing here is the schema claim.
        """
        out = tmp_path / "pd-reskin-self-train.jsonl"
        render_corpus.render_corpus(out, game_id=GAME, grading=GRADING, split="train")
        loaded = eval_training_frames.load_corpus_rows(out)
        assert len(loaded) == PD_RESKIN_TRAIN_ROWS
        assert {str(row["game_id"]) for row in loaded} == {GAME}


class TestRegradePathAgreesWithDirectRender:
    def test_a_regraded_group_mix_render_equals_a_direct_self_render(self, tmp_path: Path):
        """The two ways an arm's corpus can reach `self` grading must agree.

        Swept-then-regraded is what the twin-pd contrast does, so its selection attrition is shared
        by construction; rendering directly is what this CLI does. If the two disagreed, the
        amplification read and the trained corpus would be different prompt sets and nothing
        downstream would say so.
        """
        group_mix = tmp_path / "pd-reskin-group-mix-train.jsonl"
        regraded = tmp_path / "pd-reskin-regraded-self-train.jsonl"
        direct = tmp_path / "pd-reskin-self-train.jsonl"

        render_corpus.render_corpus(group_mix, game_id=GAME, grading="group-mix", split="train")
        regrade_corpus.main(
            ["--corpus", str(group_mix), "--grading", GRADING, "--out", str(regraded)]
        )
        render_corpus.render_corpus(direct, game_id=GAME, grading=GRADING, split="train")

        assert read_rows(regraded) == read_rows(direct)
        assert len(read_rows(direct)) == PD_RESKIN_TRAIN_ROWS

    def test_the_two_gradings_differ_in_the_grading_column_and_nothing_else(self, tmp_path: Path):
        group_mix = tmp_path / "pd-reskin-group-mix-train.jsonl"
        direct = tmp_path / "pd-reskin-self-train.jsonl"
        render_corpus.render_corpus(group_mix, game_id=GAME, grading="group-mix", split="train")
        render_corpus.render_corpus(direct, game_id=GAME, grading=GRADING, split="train")

        differing: set[str] = set()
        for before, after in zip(read_rows(group_mix), read_rows(direct), strict=True):
            assert set(before) == set(after)
            differing.update(key for key in before if before[key] != after[key])
        assert differing == {GRADING_COLUMN}
