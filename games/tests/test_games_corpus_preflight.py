"""Offline tests for `games/corpus_preflight.py`, the CPU gate a corpus passes before a card is paid for.

The failure this gate exists for happened: a 9B arm booted its engine on a rented card and died in
the reward function, because the corpus predated nine of the columns `games.rewards` rebuilds each
row from. Every one of those columns is invisible to the other consumers -- the training loader
checks two, the frames evaluator checks fourteen -- so nothing before the first reward call looked
wrong.

Which makes the sabotage case the load-bearing test here, not the happy path: `TestSabotage` deletes
each reward-schema column that no earlier consumer checks, one at a time, from every row of an
otherwise-perfect corpus, and requires the gate to go red naming that column. If that class passed
with the column check removed, this module would be a reassuring message rather than a check.

`REWARD_ONLY_COLUMNS` below is the set to sabotage: the reward columns no earlier consumer looks at,
derived from the three real column sets rather than listed, so a column added to
`games.rewards._Row` gets sabotaged here without anyone remembering this module exists.

All CPU: rendering prompts and reading JSONL touch no tokenizer, no model and no GPU.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games import corpus_preflight, render_corpus
from games.arms import ARMS
from games.eval_training_frames import REQUIRED_CORPUS_FIELDS
from games.rewards import BACKFILLABLE_REWARD_COLUMNS, REQUIRED_REWARD_COLUMNS
from games.train import CORPUS_ARM_COLUMNS, load_corpus

if TYPE_CHECKING:
    from pathlib import Path

GAME = "pd-reskin"
GRADING = "self"
ARM = "pd-reskin-self"
PD_RESKIN_TRAIN_ROWS = 176

# The backfillable set is exempt from the missing-column-goes-red requirement BY DESIGN, not by
# oversight: `games.train.load_corpus` fills each one's unset marker (the value every current row
# builder writes where the column does not apply, refused by the column's only reader), so a
# corpus swept before the column existed keeps loading. TestBackfillableColumns pins that path in
# the direction it can fail instead.
REWARD_ONLY_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        set(REQUIRED_REWARD_COLUMNS)
        - REQUIRED_CORPUS_FIELDS
        - set(CORPUS_ARM_COLUMNS)
        - set(BACKFILLABLE_REWARD_COLUMNS)
    )
)


def render(tmp_path: Path, *, game_id: str = GAME, grading: str = GRADING) -> Path:
    """Render one game's full train split, which is a corpus every consumer should accept."""
    out = tmp_path / f"{game_id}-{grading}-train.jsonl"
    render_corpus.render_corpus(out, game_id=game_id, grading=grading, split="train")
    return out


def rewrite_without(path: Path, column: str) -> Path:
    """Copy a corpus back over itself with one column deleted from every row."""
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for row in rows:
        del row[column]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


class TestPassingCorpus:
    def test_a_fresh_render_passes_and_reports_what_it_checked(self, tmp_path: Path):
        report = corpus_preflight.preflight_corpus(render(tmp_path))
        assert report.arm_name == ARM
        assert report.game_id == GAME
        assert report.grading == GRADING
        assert report.n_rows == PD_RESKIN_TRAIN_ROWS
        assert report.n_rows_after_arm_pins == PD_RESKIN_TRAIN_ROWS

    def test_the_cli_accepts_a_good_corpus(self, tmp_path: Path):
        corpus_preflight.main(["--corpus", str(render(tmp_path))])

    def test_an_explicit_arm_name_is_honoured(self, tmp_path: Path):
        report = corpus_preflight.preflight_corpus(render(tmp_path), arm_name=ARM)
        assert report.arm_name == ARM

    def test_a_payoff_pinned_arm_reports_the_rows_it_would_actually_train(self, tmp_path: Path):
        """The stag ladder's rungs pin one payoff variant each, so the pin must narrow the corpus.

        Reported separately from the file's row count because "this arm trains 128 prompts" and
        "this arm trains the 64 the pin keeps" are the difference between covering the ladder and
        covering one rung, and nothing else in the artifacts says which.
        """
        corpus = render(tmp_path, game_id="stag-hunt", grading="group-mix")
        report = corpus_preflight.preflight_corpus(corpus, arm_name="stag-hunt-safe-rung")
        assert report.n_rows_after_arm_pins < report.n_rows
        assert report.n_rows_after_arm_pins > 0


class TestSabotage:
    @pytest.mark.parametrize("column", REWARD_ONLY_COLUMNS)
    def test_deleting_one_reward_column_from_every_row_goes_red(self, tmp_path: Path, column: str):
        corpus = rewrite_without(render(tmp_path), column)
        with pytest.raises(ValueError, match=column):
            corpus_preflight.main(["--corpus", str(corpus)])

    def test_the_sabotaged_columns_are_invisible_to_every_earlier_consumer(self, tmp_path: Path):
        """Establish that the column check is the only thing catching these.

        If a sabotaged column were also checked by the training loader or the frames evaluator,
        `TestSabotage` would pass with the reward-column check deleted and prove nothing.
        """
        assert REWARD_ONLY_COLUMNS, "no reward-only columns left to sabotage"
        for column in REWARD_ONLY_COLUMNS:
            assert column not in REQUIRED_CORPUS_FIELDS
            assert column not in CORPUS_ARM_COLUMNS

    def test_one_row_missing_a_reward_column_is_caught_too(self, tmp_path: Path):
        """A corpus concatenated from two vintages is ragged rather than uniformly short."""
        corpus = render(tmp_path)
        rows: list[dict[str, Any]] = [
            json.loads(line)
            for line in corpus.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        del rows[7]["payoff_dd"]
        corpus.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="payoff_dd"):
            corpus_preflight.preflight_corpus(corpus)


class TestRefusals:
    def test_a_missing_corpus_file_names_the_path(self, tmp_path: Path):
        missing = tmp_path / "never-rendered.jsonl"
        with pytest.raises(FileNotFoundError, match=r"never-rendered\.jsonl"):
            corpus_preflight.main(["--corpus", str(missing)])

    def test_an_empty_corpus_file_is_refused(self, tmp_path: Path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no rows"):
            corpus_preflight.preflight_corpus(empty)

    def test_an_unknown_arm_name_lists_the_registered_ones(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not a registered arm"):
            corpus_preflight.preflight_corpus(render(tmp_path), arm_name="pd-reskin-selff")

    def test_an_arm_trained_on_another_game_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="but this arm trains"):
            corpus_preflight.preflight_corpus(render(tmp_path), arm_name="twin-pd-self")

    def test_an_ambiguous_game_and_grading_asks_for_an_arm(self, tmp_path: Path):
        """Three arms share stag-hunt group-mix, differing only in their payoff pin.

        Guessing one of them would report a row count for pins the run does not have, so the gate
        names the candidates and stops instead.
        """
        corpus = render(tmp_path, game_id="stag-hunt", grading="group-mix")
        with pytest.raises(ValueError, match="stag-hunt-safe-rung"):
            corpus_preflight.preflight_corpus(corpus)

    def test_a_game_no_arm_trains_is_refused(self, tmp_path: Path):
        corpus = render(tmp_path, game_id=GAME, grading="group-mix")
        with pytest.raises(ValueError, match="no registered arm"):
            corpus_preflight.preflight_corpus(corpus)

    def test_a_fresh_render_is_refused_for_an_arm_needing_a_sampled_opponent(self, tmp_path: Path):
        """`vs-fixed-mix` grades against a frozen opponent the baseline sweep has to measure first.

        A fresh render carries `opp_coop_prob = -1` on every row, which the trainer does refuse --
        but inside the reward function, after the weights have loaded and a generation batch has
        been paid for. That is the whole reason this gate exists.
        """
        corpus = render(tmp_path, game_id="pd-vs-frozen", grading="vs-fixed-mix")
        with pytest.raises(ValueError, match="opp_coop_prob"):
            corpus_preflight.preflight_corpus(corpus)


class TestBackfillableColumns:
    """The declared exception to the missing-column rule, pinned in the direction it can fail.

    A backfillable column's absence has exactly one meaning -- the corpus predates the column --
    so the loader fills the unset marker instead of refusing, and an old corpus keeps training.
    What must still hold: the marker actually lands on every row (a backfill that half-applied
    would reach the reward as a hole), and the exemption stays EXACTLY the declared set, so a
    column added to `games.rewards._Row` without a backfill entry still goes red above.
    """

    def test_a_missing_backfillable_column_preflights_green_with_the_marker_filled(
        self, tmp_path: Path
    ) -> None:
        for column, marker in BACKFILLABLE_REWARD_COLUMNS.items():
            corpus = rewrite_without(render(tmp_path), column)
            report = corpus_preflight.preflight_corpus(corpus)
            assert report.n_rows == PD_RESKIN_TRAIN_ROWS
            rows = load_corpus(str(corpus), ARMS[ARM])
            assert all(row[column] == marker for row in rows), column

    def test_every_reward_column_is_classified_exactly_once(self) -> None:
        # A column in both sets would be sabotage-tested for a red the loader prevents; one in
        # neither would silently escape the whole scheme.
        assert not set(REWARD_ONLY_COLUMNS) & set(BACKFILLABLE_REWARD_COLUMNS)
        assert set(BACKFILLABLE_REWARD_COLUMNS) <= set(REQUIRED_REWARD_COLUMNS)


class TestABreadthArmsMixedCorpus:
    """One corpus, several games, one grading: the wave-4b shape, through every consumer's own check.

    The gate's arm derivation used to read row 0's game, which on a mixed file picks an arm by
    whichever game the concatenation happened to put first, or finds none and refuses a corpus that is
    perfectly good. It now asks for an arm whose whole game set covers the file.
    """

    BREADTH_ARM = "prosocial-breadth-care1"
    BREADTH_GRADING = "care-alpha-1"

    def mixed(self, tmp_path: Path, *games: str, grading: str | None = None) -> Path:
        """Render each game's train split under one grading and concatenate them into one corpus."""
        resolved = self.BREADTH_GRADING if grading is None else grading
        lines: list[str] = []
        for game_id in games:
            rendered = render(tmp_path, game_id=game_id, grading=resolved)
            lines.extend(rendered.read_text(encoding="utf-8").splitlines())
        merged = tmp_path / "mixed.jsonl"
        merged.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        return merged

    def test_a_mixed_corpus_derives_the_breadth_arm_and_passes(self, tmp_path: Path):
        corpus = self.mixed(tmp_path, "twin-pd", "stag-hunt", "trust-vs-stated-return")
        report = corpus_preflight.preflight_corpus(corpus)
        assert report.arm_name == self.BREADTH_ARM
        assert report.game_id == "twin-pd"
        assert report.corpus_game_ids == ("stag-hunt", "trust-vs-stated-return", "twin-pd")
        assert report.n_rows_after_arm_pins == report.n_rows

    def test_the_report_names_every_game_the_file_holds(self, tmp_path: Path):
        # A report naming the arm's lead game alone would read as a single-game corpus, which is what
        # the launcher's log line would then say about a file spanning six games.
        corpus = self.mixed(tmp_path, "chicken", "public-goods")
        report = corpus_preflight.preflight_corpus(corpus)
        assert report.corpus_game_ids == ("chicken", "public-goods")

    def test_a_game_outside_the_arms_set_is_refused(self, tmp_path: Path):
        corpus = self.mixed(tmp_path, "twin-pd", "hi-lo")
        with pytest.raises(ValueError, match="hi-lo"):
            corpus_preflight.preflight_corpus(corpus, arm_name=self.BREADTH_ARM)

    def test_a_game_outside_the_set_derives_no_arm_at_all(self, tmp_path: Path):
        # Without --arm there is nothing to check the file against, because no registered arm's set
        # covers it. The refusal says so rather than picking the arm that covers the most of it.
        corpus = self.mixed(tmp_path, "twin-pd", "hi-lo")
        with pytest.raises(ValueError, match="no registered arm"):
            corpus_preflight.preflight_corpus(corpus)

    def test_a_single_game_file_of_the_arms_grading_still_derives_it(self, tmp_path: Path):
        # A breadth corpus with one game's rows is not ambiguous today, because the pair's two arms
        # differ in grading: the care weight is in the name the corpus rows carry.
        corpus = render(tmp_path, game_id="twin-pd", grading=self.BREADTH_GRADING)
        assert corpus_preflight.preflight_corpus(corpus).arm_name == self.BREADTH_ARM

    def test_the_control_arms_grading_derives_the_control(self, tmp_path: Path):
        corpus = self.mixed(tmp_path, "twin-pd", "chicken", grading="care-alpha-0")
        assert corpus_preflight.preflight_corpus(corpus).arm_name == "prosocial-breadth-self"
