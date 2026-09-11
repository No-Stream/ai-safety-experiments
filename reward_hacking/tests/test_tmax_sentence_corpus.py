"""The sentence corpus writer: the 290 pairs as five-key rows named from `stimuli`, read back by the capture loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from games.interp_cells import STIMULUS_RENDER_TEMPLATED, load_stimuli, stimuli_digest
from reward_hacking.interp.stimuli import (
    CONCEPT_CONTRADICTION,
    CONCEPT_DECEPTION,
    CONCEPT_EVAL_AWARENESS,
    CONCEPT_SHORTCUT,
    CONCEPTS,
    SENTENCE_SIDE_NEGATIVE,
    SENTENCE_SIDE_POSITIVE,
    SENTENCE_SIDES,
    SENTENCE_STIMULUS_SETS,
)
from reward_hacking.interp.tmax_sentence_corpus import (
    STIMULI_FILENAME,
    STIMULUS_RENDER,
    SentenceCorpusError,
    main,
    sentence_stimuli,
    write_sentence_stimuli,
)

if TYPE_CHECKING:
    from pathlib import Path

N_PAIRS = 290


class TestSentenceStimuli:
    def test_the_four_sets_hold_the_290_pairs_under_the_stimuli_names(self) -> None:
        rows = sentence_stimuli()
        assert len(rows) == 2 * N_PAIRS
        assert sum(len(pairs) for pairs in SENTENCE_STIMULUS_SETS.values()) == N_PAIRS
        assert set(SENTENCE_STIMULUS_SETS) == {
            CONCEPT_SHORTCUT,
            CONCEPT_DECEPTION,
            CONCEPT_EVAL_AWARENESS,
            CONCEPT_CONTRADICTION,
        }
        assert set(CONCEPTS) == set(SENTENCE_STIMULUS_SETS) - {CONCEPT_CONTRADICTION}
        assert {row.stimulus_set for row in rows} == set(SENTENCE_STIMULUS_SETS)
        assert {row.side for row in rows} == set(SENTENCE_SIDES)
        assert len({row.stimulus_id for row in rows}) == len(rows)
        assert all(row.assistant_prefix is None for row in rows)

    def test_every_pair_has_one_row_per_side_carrying_that_pairs_sentence(self) -> None:
        rows = sentence_stimuli([CONCEPT_EVAL_AWARENESS])
        by_pair: dict[str, dict[str, str]] = {}
        for row in rows:
            by_pair.setdefault(row.pair_id, {})[row.side] = row.text
        pairs = SENTENCE_STIMULUS_SETS[CONCEPT_EVAL_AWARENESS]
        assert len(by_pair) == len(pairs)
        for index, (pair_id, sides) in enumerate(by_pair.items()):
            assert pair_id == f"{CONCEPT_EVAL_AWARENESS}--{index:03d}"
            assert sides == {
                SENTENCE_SIDE_POSITIVE: pairs[index].positive,
                SENTENCE_SIDE_NEGATIVE: pairs[index].negative,
            }

    def test_an_unknown_set_is_refused_with_the_known_names_listed(self) -> None:
        with pytest.raises(
            SentenceCorpusError,
            match=r"\['eval-awareness'\] are not sentence stimulus sets; the sets are "
            r"\['contradiction', 'deception', 'eval_awareness', 'shortcut'\]",
        ):
            sentence_stimuli(["eval-awareness"])

    def test_the_corpus_is_authored_for_templating_here(self) -> None:
        assert STIMULUS_RENDER == STIMULUS_RENDER_TEMPLATED


class TestWriteSentenceStimuli:
    def test_the_file_reads_back_through_the_capture_loader_with_a_stable_digest(
        self, tmp_path: Path
    ) -> None:
        path = write_sentence_stimuli(tmp_path)
        assert path == tmp_path / STIMULI_FILENAME
        loaded = load_stimuli(path)
        assert loaded == sentence_stimuli()
        assert stimuli_digest(loaded) == stimuli_digest(sentence_stimuli())
        assert [row.stimulus_set for row in loaded[:2]] == [CONCEPT_SHORTCUT, CONCEPT_SHORTCUT]

    def test_a_subset_writes_only_those_sets_in_the_order_asked(self, tmp_path: Path) -> None:
        assert main(["--out-dir", str(tmp_path), "--sets", "eval_awareness,contradiction"]) == 0
        loaded = load_stimuli(tmp_path / STIMULI_FILENAME)
        sets_in_order = list(dict.fromkeys(row.stimulus_set for row in loaded))
        assert sets_in_order == [CONCEPT_EVAL_AWARENESS, CONCEPT_CONTRADICTION]

    def test_refuses_to_overwrite(self, tmp_path: Path) -> None:
        write_sentence_stimuli(tmp_path)
        with pytest.raises(FileExistsError):
            write_sentence_stimuli(tmp_path)
