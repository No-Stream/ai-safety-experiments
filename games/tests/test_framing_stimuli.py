"""The framing capture corpus: twin-pd eval prompts under every counterpart framing.

Leg A of the framing x interp cross reads where each framing's prompts sit along the base-fitted
lead direction, so the corpus these tests pin is the join surface of that whole read: `side` must
be the framing id (the grouping key), `pair_id` must be shared by exactly the registered framings of
one underlying game row (the within-prompt contrast key), and the twin rows must be byte-identical
to the trained rendering (the reference cell). A corpus violating any of these would not error in
the capture -- it would produce well-formed tensors grouped wrongly.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from games.framing_stimuli import (
    FRAMING_CAPTURE_SET,
    framing_capture_stimuli,
    write_framing_stimuli,
)
from games.interp_cells import load_stimuli
from games.prompts import (
    COUNTERPART_FRAMING_IDS,
    FRAMING_TWIN,
    LABEL_PRINT_ORDERS,
    SPLIT_EVAL,
    generate_prompt_rows,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestFramingCaptureCorpus:
    def test_the_corpus_has_the_designed_arithmetic(self) -> None:
        """Every framing x 4 eval frames x 2 payoff variants x 2 mappings x 2 orders."""
        stimuli = framing_capture_stimuli()
        assert len(stimuli) == len(COUNTERPART_FRAMING_IDS) * 16 * len(LABEL_PRINT_ORDERS)
        assert {stimulus.stimulus_set for stimulus in stimuli} == {FRAMING_CAPTURE_SET}
        assert len({stimulus.stimulus_id for stimulus in stimuli}) == len(stimuli)

    def test_side_is_the_framing_and_every_pair_holds_every_framing(self) -> None:
        stimuli = framing_capture_stimuli()
        assert {stimulus.side for stimulus in stimuli} == set(COUNTERPART_FRAMING_IDS)
        by_pair: dict[str, set[str]] = {}
        for stimulus in stimuli:
            by_pair.setdefault(stimulus.pair_id, set()).add(stimulus.side)
        assert all(sides == set(COUNTERPART_FRAMING_IDS) for sides in by_pair.values())
        assert len(by_pair) == 16 * len(LABEL_PRINT_ORDERS)

    def test_pair_ids_never_leak_a_framing_segment(self) -> None:
        """A pair_id carrying `--framing-` would silently split one row's framings apart."""
        stimuli = framing_capture_stimuli()
        assert all("--framing-" not in stimulus.pair_id for stimulus in stimuli)

    def test_twin_rows_are_byte_identical_to_the_trained_rendering(self) -> None:
        """The reference framing must read exactly what the trained arms read."""
        stimuli = framing_capture_stimuli()
        twin_texts = {
            stimulus.pair_id: stimulus.text for stimulus in stimuli if stimulus.side == FRAMING_TWIN
        }
        for order in LABEL_PRINT_ORDERS:
            for row in generate_prompt_rows(
                "twin-pd", "self", split=SPLIT_EVAL, label_print_order=order
            ):
                assert twin_texts[str(row["prompt_id"])] == str(row["prompt"])

    def test_no_stimulus_carries_an_assistant_prefix(self) -> None:
        """Prompt-end pooling is the read; a teacher-forced continuation would move it."""
        stimuli = framing_capture_stimuli()
        assert all(stimulus.assistant_prefix is None for stimulus in stimuli)


class TestWriter:
    def test_the_written_corpus_round_trips_through_the_capture_loader(
        self, tmp_path: Path
    ) -> None:
        stimuli_path, provenance_path = write_framing_stimuli(tmp_path)
        loaded = load_stimuli(stimuli_path)
        built = framing_capture_stimuli()
        assert loaded == built
        provenance_lines = provenance_path.read_text().splitlines()
        assert len(provenance_lines) == len(built)

    def test_provenance_carries_the_join_fields(self, tmp_path: Path) -> None:
        _, provenance_path = write_framing_stimuli(tmp_path)
        first = json.loads(provenance_path.read_text().splitlines()[0])
        assert {"id", "framing", "label_print_order", "coop_label", "payoff_variant"} <= set(first)


class TestRefusals:
    def test_an_unknown_framing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown framing_id"):
            framing_capture_stimuli(framings=("martian",))
