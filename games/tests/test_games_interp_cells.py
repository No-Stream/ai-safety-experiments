"""Pin the guards that decide whether two cached captures may be compared at all.

Offline and CPU-only: every cell here is a handful of small tensors written to a tmp_path, because
each guard is about *identity* rather than about numbers. All of them exist because their failure
mode is a plausible result rather than a crash.

:class:`TestStimuliDigestGuard` is the sabotage this module was written around. An analysis reads a
stimulus corpus for its pair structure and reads tensors from a capture root, and nothing in either
file connects them: every stimulus id still resolves after a corpus is edited, so the numbers stay
plausible and answer a question about text that changed. `test_a_mismatched_corpus_is_refused` plants
exactly that mismatch, and `test_the_digest_moves_with_the_text` and
`test_a_truncated_corpus_is_a_different_corpus` show the digest is what makes it detectable.

:class:`TestLadderIdentity` is the moved-apparatus trap: a batch size, a storage dtype or a layer
convention differing between two cells changes the activations by at least as much as a trained LoRA
delta does, so a before/after read across them measures the capture, not the model.

:class:`TestRowAlignment` is the reordered-rows trap. Every read is row-wise, so a cell holding the
same ids in a different order yields a number rather than an error.

:class:`TestAdapterMovedActivations` is the inert-adapter trap: an adapter can load cleanly and then
not reach the forward pass, which leaves base activations under a trained checkpoint's name.

:class:`TestPairLayout` and :class:`TestStorage` cover the two remaining silent losses -- a half pair
narrowing a denominator without saying so, and float16 mapping a late-layer coordinate to infinity.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from games.interp_cells import (
    ACTIVATIONS_FILENAME,
    BASE_ARM,
    BASE_STEP,
    CELL_MANIFEST_FILENAME,
    CellFormatError,
    CellIdentity,
    Stimulus,
    StimulusFileError,
    assert_adapter_moved_activations,
    assert_ladder_identity,
    assert_rows_align,
    assert_stimuli_match,
    concept_activations,
    load_ladder,
    load_stimuli,
    pair_layout,
    read_cell,
    row_index_for,
    step_dir,
    stimuli_digest,
    write_cell,
)

N_LAYERS = 3
HIDDEN = 4
N_PAIRS = 4


def make_stimuli(n_pairs: int = N_PAIRS, *, stimulus_set: str = "axis-a") -> list[Stimulus]:
    """Two-sided stimuli for one set, in the A, B, A, B order the corpus writer emits."""
    return [
        Stimulus(
            stimulus_id=f"{stimulus_set}--p{index}--{side}",
            stimulus_set=stimulus_set,
            side=side,
            pair_id=f"{stimulus_set}--p{index}",
            text=f"rendered text for pair {index} side {side}",
        )
        for index in range(n_pairs)
        for side in ("A", "B")
    ]


def make_identity(digest: str, **overrides: Any) -> CellIdentity:
    """A cell identity with the fields a test is not varying held fixed."""
    fields: dict[str, Any] = {
        "base_model": "Qwen/Qwen3.5-2B",
        "stimuli_sha256": digest,
        "rendered_sha256": f"rendered-{digest}",
        "layer_convention": "post_block",
        "n_layers": N_LAYERS,
        "hidden_size": HIDDEN,
        "batch_size": 1,
        "compute_dtype": "bfloat16",
        "store_dtype": "float32",
        "stimulus_render": "verbatim",
    }
    fields.update(overrides)
    return CellIdentity(**fields)


def write_test_cell(  # noqa: PLR0913 - a fixture cell is a root, an arm, a step, stimuli and an identity
    root: Path,
    arm: str,
    step: int,
    stimuli: list[Stimulus],
    identity: CellIdentity,
    *,
    offset: float = 0.0,
    applied: int | None = 7,
) -> Path:
    """Write one cell whose activations separate the two sides by `offset` along dimension 0."""
    matrix = torch.zeros(len(stimuli), identity.n_layers, identity.hidden_size)
    for row, stimulus in enumerate(stimuli):
        matrix[row, :, 0] = offset if stimulus.side == "A" else -offset
        matrix[row, :, 1] = float(row) * 0.01
    stimulus_set = stimuli[0].stimulus_set
    return write_cell(
        step_dir(root, arm, step),
        arm=arm,
        step=step,
        identity=identity,
        rows={stimulus_set: row_index_for(stimuli, [11] * len(stimuli))},
        activations={(stimulus_set, "mean"): matrix},
        applied_adapter_weights=None if arm == BASE_ARM else applied,
        adapter_weights_sha256=None if arm == BASE_ARM else "deadbeef",
        provenance={"git_sha": "testing"},
    ).parent


class TestStimuliDigestGuard:
    def test_a_mismatched_corpus_is_refused(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        digest = stimuli_digest(stimuli)
        write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, make_identity(digest))
        cells = [read_cell(step_dir(tmp_path, BASE_ARM, BASE_STEP))]

        assert_stimuli_match(cells, digest)

        edited = [*stimuli[:-1], Stimulus(**{**vars(stimuli[-1]), "text": "a different rendering"})]
        with pytest.raises(CellFormatError, match="different stimulus corpus"):
            assert_stimuli_match(cells, stimuli_digest(edited))

    def test_load_ladder_refuses_a_mismatched_corpus(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        write_test_cell(
            tmp_path, BASE_ARM, BASE_STEP, stimuli, make_identity(stimuli_digest(stimuli))
        )
        with pytest.raises(CellFormatError, match="different stimulus corpus"):
            load_ladder(tmp_path, stimuli_sha256="0" * 64)

    def test_the_digest_moves_with_the_text(self) -> None:
        stimuli = make_stimuli()
        edited = [*stimuli[:-1], Stimulus(**{**vars(stimuli[-1]), "text": "changed"})]
        assert stimuli_digest(stimuli) != stimuli_digest(edited)

    def test_the_digest_moves_with_the_order(self) -> None:
        stimuli = make_stimuli()
        assert stimuli_digest(stimuli) != stimuli_digest([stimuli[1], stimuli[0], *stimuli[2:]])

    def test_a_truncated_corpus_is_a_different_corpus(self) -> None:
        """A smoke over the first few rows must not pass as a thin version of the full corpus."""
        stimuli = make_stimuli()
        assert stimuli_digest(stimuli[:4]) != stimuli_digest(stimuli)

    def test_a_boundary_shift_between_fields_does_not_collide(self) -> None:
        """Length-delimiting is what stops 'ab' + 'c' hashing the same as 'a' + 'bc'."""
        left = [Stimulus("ab", "s", "A", "p", "c")]
        right = [Stimulus("a", "s", "A", "p", "bc")]
        assert stimuli_digest(left) != stimuli_digest(right)


class TestLadderIdentity:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("batch_size", 4),
            ("rendered_sha256", "a-different-rendering"),
            ("store_dtype", "float16"),
            ("compute_dtype", "float32"),
            ("layer_convention", "hidden_states"),
            ("base_model", "Qwen/Qwen3.5-4B"),
            ("stimulus_render", "templated_here"),
        ],
    )
    def test_a_moved_apparatus_is_refused(self, tmp_path: Path, field: str, value: object) -> None:
        stimuli = make_stimuli()
        digest = stimuli_digest(stimuli)
        write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, make_identity(digest))
        write_test_cell(
            tmp_path, "arm", 10, stimuli, make_identity(digest, **{field: value}), offset=0.5
        )
        with pytest.raises(CellFormatError, match="did not measure the same thing"):
            load_ladder(tmp_path)

    def test_every_differing_field_is_reported_at_once(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        digest = stimuli_digest(stimuli)
        cells = [
            read_cell(
                write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, make_identity(digest))
            ),
            read_cell(
                write_test_cell(
                    tmp_path,
                    "arm",
                    10,
                    stimuli,
                    make_identity(digest, batch_size=4, store_dtype="float16"),
                    offset=0.5,
                )
            ),
        ]
        with pytest.raises(CellFormatError) as error:
            assert_ladder_identity(cells)
        assert "batch_size" in str(error.value)
        assert "store_dtype" in str(error.value)

    def test_provenance_may_differ(self, tmp_path: Path) -> None:
        """A git sha or a timestamp differing is a fact about when work ran, not what it measured."""
        stimuli = make_stimuli()
        digest = stimuli_digest(stimuli)
        first = read_cell(
            write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, make_identity(digest))
        )
        second_dir = write_test_cell(
            tmp_path, "arm", 10, stimuli, make_identity(digest), offset=0.5
        )
        manifest_path = second_dir / CELL_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"]["git_sha"] = "a-different-commit"
        manifest_path.write_text(json.dumps(manifest))
        assert_ladder_identity([first, read_cell(second_dir)])


class TestRowAlignment:
    def test_reordered_rows_are_refused(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        digest = stimuli_digest(stimuli)
        identity = make_identity(digest)
        write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity)
        write_test_cell(
            tmp_path, "arm", 10, [stimuli[1], stimuli[0], *stimuli[2:]], identity, offset=0.5
        )
        with pytest.raises(CellFormatError, match="same rows"):
            load_ladder(tmp_path)

    def test_a_missing_row_names_what_is_missing(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        digest = stimuli_digest(stimuli)
        identity = make_identity(digest)
        first = read_cell(write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity))
        second = read_cell(write_test_cell(tmp_path, "arm", 10, stimuli[:-2], identity, offset=0.5))
        with pytest.raises(CellFormatError, match=stimuli[-1].stimulus_id):
            assert_rows_align([first, second])

    def test_a_cell_missing_a_pooling_is_refused(self, tmp_path: Path) -> None:
        """A read present on one side of a comparison and absent on the other narrows it silently."""
        stimuli = make_stimuli()
        digest = stimuli_digest(stimuli)
        identity = make_identity(digest)
        first = read_cell(write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity))
        matrix = torch.zeros(len(stimuli), N_LAYERS, HIDDEN)
        matrix[:, :, 0] = 0.5
        second_dir = step_dir(tmp_path, "arm", 10)
        write_cell(
            second_dir,
            arm="arm",
            step=10,
            identity=identity,
            rows={"axis-a": row_index_for(stimuli, [11] * len(stimuli))},
            activations={("axis-a", "mean"): matrix, ("axis-a", "last"): matrix + 0.1},
            applied_adapter_weights=7,
            adapter_weights_sha256="deadbeef",
            provenance={},
        )
        with pytest.raises(CellFormatError, match="holds poolings"):
            assert_rows_align([first, read_cell(second_dir)])

    def test_a_swapped_side_is_refused(self, tmp_path: Path) -> None:
        """Same ids, same order, one row's side flipped: the pairing differs and the diff inverts."""
        stimuli = make_stimuli()
        digest = stimuli_digest(stimuli)
        identity = make_identity(digest)
        first = read_cell(write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity))
        flipped = [Stimulus(**{**vars(stimuli[0]), "side": "B"}), *stimuli[1:]]
        second = read_cell(write_test_cell(tmp_path, "arm", 10, flipped, identity, offset=0.5))
        with pytest.raises(CellFormatError, match="not on their sides"):
            assert_rows_align([first, second])


class TestAdapterMovedActivations:
    def test_an_inert_adapter_is_refused(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity, offset=0.25)
        write_test_cell(tmp_path, "arm", 10, stimuli, identity, offset=0.25, applied=372)
        with pytest.raises(CellFormatError, match="bit-identical to the base"):
            load_ladder(tmp_path)

    def test_a_moved_adapter_passes(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity, offset=0.25)
        write_test_cell(tmp_path, "arm", 10, stimuli, identity, offset=0.5, applied=372)
        ladder = load_ladder(tmp_path)
        assert [cell.label for cell in ladder.cells] == ["arm/step-10", "base/step-0"]

    def test_without_a_base_cell_the_check_is_skipped(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        first = read_cell(write_test_cell(tmp_path, "arm", 10, stimuli, identity, offset=0.25))
        second = read_cell(write_test_cell(tmp_path, "arm", 20, stimuli, identity, offset=0.25))
        assert_adapter_moved_activations([first, second])


class TestLadderFilters:
    def test_an_absent_step_is_fatal(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity)
        write_test_cell(tmp_path, "arm", 10, stimuli, identity, offset=0.5)
        with pytest.raises(CellFormatError, match=r"no cells for steps \[20\]"):
            load_ladder(tmp_path, steps=[10, 20])

    def test_an_absent_arm_is_fatal(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity)
        write_test_cell(tmp_path, "arm", 10, stimuli, identity, offset=0.5)
        with pytest.raises(CellFormatError, match="no cells for arms"):
            load_ladder(tmp_path, arms=["missing-arm"])

    def test_the_base_cell_survives_an_arm_filter(self, tmp_path: Path) -> None:
        """Filtering to one arm must keep the shared anchor, or every drift read loses its zero."""
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity)
        write_test_cell(tmp_path, "arm", 10, stimuli, identity, offset=0.5)
        write_test_cell(tmp_path, "other", 10, stimuli, identity, offset=0.75)
        ladder = load_ladder(tmp_path, arms=["arm"])
        assert sorted(ladder.arms) == ["arm", "base"]


class TestCellRoundTrip:
    def test_a_manifest_from_a_different_run_is_refused(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        cell_dir = write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity)
        manifest_path = cell_dir / CELL_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["sets"]["axis-a"]
        entry["stimulus_ids"] = entry["stimulus_ids"][:-1]
        entry["sides"] = entry["sides"][:-1]
        entry["pair_ids"] = entry["pair_ids"][:-1]
        entry["token_counts"] = entry["token_counts"][:-1]
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(CellFormatError, match="written by different runs"):
            read_cell(cell_dir)

    def test_an_identity_block_missing_a_field_names_the_file(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        cell_dir = write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity)
        manifest_path = cell_dir / CELL_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        del manifest["identity"]["batch_size"]
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(CellFormatError, match="batch_size"):
            read_cell(cell_dir)

    def test_an_unfinished_cell_is_not_a_cell(self, tmp_path: Path) -> None:
        """The manifest is written last, so a cell without one is work in flight, not a result."""
        cell_dir = step_dir(tmp_path, BASE_ARM, BASE_STEP)
        cell_dir.mkdir(parents=True)
        (cell_dir / ACTIVATIONS_FILENAME).write_bytes(b"")
        with pytest.raises(CellFormatError, match="not a finished cell"):
            read_cell(cell_dir)

    def test_a_layer_outside_the_capture_is_refused(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        cell = read_cell(write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity))
        with pytest.raises(CellFormatError, match="outside this 3-layer capture"):
            cell.layer("axis-a", "mean", N_LAYERS)


class TestStorage:
    def test_float16_overflow_is_refused_rather_than_stored_as_infinity(
        self, tmp_path: Path
    ) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli), store_dtype="float16")
        matrix = torch.zeros(len(stimuli), N_LAYERS, HIDDEN)
        matrix[0, -1, 0] = 70000.0
        with pytest.raises(CellFormatError, match="non-finite"):
            write_cell(
                step_dir(tmp_path, BASE_ARM, BASE_STEP),
                arm=BASE_ARM,
                step=BASE_STEP,
                identity=identity,
                rows={"axis-a": row_index_for(stimuli, [11] * len(stimuli))},
                activations={("axis-a", "mean"): matrix},
                applied_adapter_weights=None,
                adapter_weights_sha256=None,
                provenance={},
            )


class TestPairLayout:
    def test_pairs_are_ordered_by_first_appearance(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        cell = read_cell(write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity))
        layout = pair_layout(cell, "axis-a", positive_side="A")
        assert layout.pair_ids == tuple(f"axis-a--p{index}" for index in range(N_PAIRS))
        assert layout.positive_rows.tolist() == [0, 2, 4, 6]
        assert layout.negative_rows.tolist() == [1, 3, 5, 7]
        assert layout.half(0).tolist() == [0, 2]
        assert layout.half(1).tolist() == [1, 3]

    def test_a_half_pair_is_refused(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        cell = read_cell(write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli[:-1], identity))
        with pytest.raises(CellFormatError, match="missing a side"):
            pair_layout(cell, "axis-a", positive_side="A")

    def test_an_unknown_positive_side_is_refused(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        cell = read_cell(write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity))
        with pytest.raises(CellFormatError, match="positive side 'conflicting'"):
            pair_layout(cell, "axis-a", positive_side="conflicting")

    def test_concept_activations_carry_the_planted_separation(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        cell = read_cell(
            write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity, offset=0.75)
        )
        layout = pair_layout(cell, "axis-a", positive_side="A")
        concept = concept_activations(cell, "axis-a", "mean", 1, layout)
        assert concept.n_pairs == N_PAIRS
        assert concept.diff_of_means()[0].item() == pytest.approx(1.5)

    def test_a_pair_subset_restricts_the_read(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        identity = make_identity(stimuli_digest(stimuli))
        cell = read_cell(write_test_cell(tmp_path, BASE_ARM, BASE_STEP, stimuli, identity))
        layout = pair_layout(cell, "axis-a", positive_side="A")
        concept = concept_activations(cell, "axis-a", "mean", 0, layout, pairs=layout.half(1))
        assert concept.n_pairs == 2


class TestStimulusCorpus:
    def write_rows(self, rows: list[dict[str, str]], *, tmp: Path | None = None) -> Path:
        path = (tmp or Path(tempfile.mkdtemp())) / "stimuli.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return path

    def test_an_assistant_prefix_is_carried(self) -> None:
        """The corpus writer emits one per row: the reasoning to teacher-force after `<think>`."""
        path = self.write_rows(
            [
                {
                    "id": "one",
                    "set": "s",
                    "side": "A",
                    "pair_id": "p",
                    "text": "hi",
                    "assistant_prefix": "so",
                }
            ]
        )
        assert load_stimuli(path)[0].assistant_prefix == "so"

    def test_an_unknown_field_is_still_refused(self) -> None:
        path = self.write_rows(
            [{"id": "one", "set": "s", "side": "A", "pair_id": "p", "text": "hi", "sytle": "x"}]
        )
        with pytest.raises(StimulusFileError, match="unrecognised fields"):
            load_stimuli(path)

    def test_the_digest_covers_the_assistant_prefix(self) -> None:
        """The prefix is where the contrast lives, so two sides differing only there must not collide."""
        base = Stimulus("one", "s", "A", "p", "shared stem", "commits to PATCH")
        other = Stimulus("one", "s", "B", "p", "shared stem", "commits to SNAPSHOT")
        assert stimuli_digest([base]) != stimuli_digest([other])

    def test_a_duplicate_id_names_both_lines(self, tmp_path: Path) -> None:
        row = {"id": "one", "set": "s", "side": "A", "pair_id": "p", "text": "hello"}
        path = self.write_rows([row, {**row, "side": "B"}], tmp=tmp_path)
        with pytest.raises(StimulusFileError, match="repeats id"):
            load_stimuli(path)

    def test_empty_text_is_refused(self, tmp_path: Path) -> None:
        path = self.write_rows(
            [{"id": "one", "set": "s", "side": "A", "pair_id": "p", "text": "  "}], tmp=tmp_path
        )
        with pytest.raises(StimulusFileError, match="empty text"):
            load_stimuli(path)

    def test_a_round_trip_keeps_order_and_fields(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()
        path = tmp_path / "stimuli.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": stimulus.stimulus_id,
                        "set": stimulus.stimulus_set,
                        "side": stimulus.side,
                        "pair_id": stimulus.pair_id,
                        "text": stimulus.text,
                    }
                )
                for stimulus in stimuli
            )
            + "\n"
        )
        assert load_stimuli(path) == stimuli
        assert stimuli_digest(load_stimuli(path)) == stimuli_digest(stimuli)
