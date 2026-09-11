"""The contracts between what the TMAX capture driver WRITES and what the CPU readers READ, round-tripped.

Five Phase 0 modules were built in parallel and each spelled its side of three contracts alone: the
mirror cell's `replica_of` provenance, the sentence pairs' set and side names, and the twin-corpus
sidecar. A mismatch in any of them costs a GPU box a capture nobody can read, so every test here goes
through the real writer code path and the real reader:

* :class:`TestSidecarContract` builds a sidecar through `tmax_twin_sidecar.SidecarStimulus` and
  `sidecar_payload` (what `tmax_twin_corpus.write_corpus` serialises) and loads it through
  `load_span_sidecar` (what the ladder reads); an entry missing `spans_absent`, an entry filed under
  the wrong side and another set's name are each refused by name rather than defaulted.
* :class:`TestWriterReaderContracts` writes cells through `tmax_capture_ladder.write_full_weights_cell`
  and reads them back through `load_full_weights_ladder`, `declared_replica_anchor`,
  `assert_replica_displacement_zero`, `coherence_of` and `spans_absent_of`: the mirror's `replica_of`,
  stored as the CLI label `base:0`, resolves to the base cell's games label `base/step-0` and the
  displacement sabotage reads exactly 0.0; a manifest with the key renamed is refused by name.
* :class:`TestSentenceContract` pins the sentence corpus to the readers: the rows `tmax_sentence_corpus`
  writes template under `templated_here` and are refused under `verbatim`; a cell holding them under
  the written set name yields `e` through `sentence_directions`; a misspelled set is refused with the
  known names listed; a cell that stored the set under another name is refused at load.

The tiny composite model, char tokenizer and synthetic cells come from the two sibling test modules.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from games.interp_capture import render_stimuli
from games.interp_cells import (
    CELL_MANIFEST_FILENAME,
    STIMULUS_RENDER_TEMPLATED,
    STIMULUS_RENDER_VERBATIM,
    CellFormatError,
    RowIndex,
    Stimulus,
    read_cell,
    row_index_for,
    step_dir,
)
from reward_hacking.interp.stimuli import (
    CONCEPT_EVAL_AWARENESS,
    SENTENCE_SIDE_POSITIVE,
    SENTENCE_STIMULUS_SETS,
)
from reward_hacking.interp.tmax_capture_ladder import write_full_weights_cell
from reward_hacking.interp.tmax_directions import (
    NAME_EVAL_AWARENESS,
    STRATUM_ALL,
    DirectionRefusalError,
    sentence_directions,
)
from reward_hacking.interp.tmax_displacement_series import (
    ReplicaDisplacementError,
    assert_replica_displacement_zero,
)
from reward_hacking.interp.tmax_full_weights import (
    COHERENCE_FIELD,
    REPLICA_OF_FIELD,
    SPANS_ABSENT_FIELD,
    CaptureRunFacts,
    CoherenceTriple,
    LoadingReport,
    coherence_of,
    declared_replica_anchor,
    load_full_weights_ladder,
    spans_absent_of,
)
from reward_hacking.interp.tmax_sentence_corpus import sentence_stimuli
from reward_hacking.interp.tmax_twin_sidecar import (
    ENTRY_SPANS_ABSENT_KEY,
    ENTRY_SPANS_KEY,
    PROBLEM_ID_KEY,
    PROBLEM_PERTURBATION_KEY,
    PROBLEM_STIMULI_KEY,
    SIDE_HONEST,
    SIDE_RIGGED,
    SIDECAR_PROBLEMS_KEY,
    SIDECAR_SET_KEY,
    SidecarStimulus,
    SpanTable,
    TwinSidecarError,
    detectable_by_problem,
    load_span_sidecar,
    sidecar_payload,
)
from reward_hacking.tests.test_interp_tmax_capture_ladder import (
    FUSED_KERNELS,
    TURN_PREFIX,
    TURN_SUFFIX,
    TinyTokenizer,
    facts_with,
    hub_spec,
    identity,
    span_table_for,
    verbatim_stimuli,
)
from reward_hacking.tests.test_interp_tmax_capture_ladder import (
    HIDDEN as TINY_HIDDEN,
)
from reward_hacking.tests.test_interp_tmax_capture_ladder import (
    N_LAYERS as TINY_LAYERS,
)
from reward_hacking.tests.test_tmax_direction_readers import (
    FAST,
    HIDDEN,
    N_LAYERS,
    PLANTED_LAYER,
    UNIDENTIFIABLE,
    axis,
    cell_from,
    read_at,
)

if TYPE_CHECKING:
    from pathlib import Path

    from games.interp_cells import CapturedCell


# --------------------------------------------------------------------------------------
# The sidecar: written through the record type, read through the capture's loader
# --------------------------------------------------------------------------------------


def sidecar_file(
    tmp_path: Path, stimuli: list[Stimulus], table: SpanTable
) -> tuple[Path, dict[str, Any]]:
    """Write a sidecar as the corpus does: entries via `SidecarStimulus`, the top level via `sidecar_payload`."""
    entries: dict[str, dict[str, Any]] = {
        stimulus.side: SidecarStimulus(
            stimulus_id=stimulus.stimulus_id,
            side=stimulus.side,
            input_ids=table.input_ids[stimulus.stimulus_id],
            spans=table.spans[stimulus.stimulus_id],
            spans_absent=table.spans_absent[stimulus.stimulus_id],
        ).to_dict()
        for stimulus in stimuli
    }
    sidecar: dict[str, Any] = sidecar_payload(
        stimuli_sha256="digest",
        problems=[
            {
                PROBLEM_ID_KEY: "p0",
                PROBLEM_PERTURBATION_KEY: {"detectable": True},
                PROBLEM_STIMULI_KEY: entries,
            }
        ],
        base_model="tiny",
    )
    path = tmp_path / "twin-corpus.json"
    path.write_text(json.dumps(sidecar))
    return path, sidecar


class TestSidecarContract:
    def test_a_sidecar_written_through_the_record_type_loads_through_the_capture_reader(
        self, tmp_path: Path
    ) -> None:
        stimuli = verbatim_stimuli(n_pairs=1)
        table = span_table_for(TinyTokenizer(), stimuli, drop_head_from="p0--honest-inline")
        path, _ = sidecar_file(tmp_path, stimuli, table)
        loaded = load_span_sidecar(path)
        assert loaded.spans == table.spans
        assert loaded.spans_absent == table.spans_absent
        assert loaded.input_ids == table.input_ids
        assert loaded.span_names == ("full", "head")
        assert loaded.stimulus_render == STIMULUS_RENDER_VERBATIM
        assert detectable_by_problem(path) == {"p0": True}

    def test_a_span_outside_the_prompt_is_refused(self, tmp_path: Path) -> None:
        stimuli = verbatim_stimuli(n_pairs=1)
        path, sidecar = sidecar_file(tmp_path, stimuli, span_table_for(TinyTokenizer(), stimuli))
        entries = sidecar[SIDECAR_PROBLEMS_KEY][0][PROBLEM_STIMULI_KEY]
        entries[SIDE_RIGGED][ENTRY_SPANS_KEY]["head"] = [0, 10_000]
        path.write_text(json.dumps(sidecar))
        with pytest.raises(TwinSidecarError, match="non-empty window inside the prompt"):
            load_span_sidecar(path)

    def test_an_entry_missing_spans_absent_is_refused_by_name_not_defaulted(
        self, tmp_path: Path
    ) -> None:
        stimuli = verbatim_stimuli(n_pairs=1)
        path, sidecar = sidecar_file(tmp_path, stimuli, span_table_for(TinyTokenizer(), stimuli))
        entries = sidecar[SIDECAR_PROBLEMS_KEY][0][PROBLEM_STIMULI_KEY]
        del entries[SIDE_HONEST][ENTRY_SPANS_ABSENT_KEY]
        path.write_text(json.dumps(sidecar))
        with pytest.raises(TwinSidecarError, match=r"lacks \['spans_absent'\]"):
            load_span_sidecar(path)

    def test_an_entry_filed_under_the_wrong_side_is_refused(self, tmp_path: Path) -> None:
        stimuli = verbatim_stimuli(n_pairs=1)
        path, sidecar = sidecar_file(tmp_path, stimuli, span_table_for(TinyTokenizer(), stimuli))
        entries = sidecar[SIDECAR_PROBLEMS_KEY][0][PROBLEM_STIMULI_KEY]
        entries[SIDE_RIGGED], entries[SIDE_HONEST] = entries[SIDE_HONEST], entries[SIDE_RIGGED]
        path.write_text(json.dumps(sidecar))
        with pytest.raises(TwinSidecarError, match="files stimulus"):
            load_span_sidecar(path)

    def test_another_sets_sidecar_is_refused(self, tmp_path: Path) -> None:
        stimuli = verbatim_stimuli(n_pairs=1)
        path, sidecar = sidecar_file(tmp_path, stimuli, span_table_for(TinyTokenizer(), stimuli))
        sidecar[SIDECAR_SET_KEY] = "tmax-grader-twins-v2"
        path.write_text(json.dumps(sidecar))
        with pytest.raises(TwinSidecarError, match="names stimulus set"):
            load_span_sidecar(path)


# --------------------------------------------------------------------------------------
# The provenance block: written by the ladder, read by the CPU modules
# --------------------------------------------------------------------------------------

RUN_FACTS = CaptureRunFacts(
    git_sha="deadbeef",
    torch_version="0.0",
    device="cpu",
    stimuli_file="stimuli.jsonl",
    spans_sidecar="twin-corpus.json",
    span_names=("head",),
    tokenizer_id="tiny",
    tokenizer_revision="rev",
)
COHERENT = CoherenceTriple(0.9, 0.04, 0.97, source="phase1/summary.json", reason=None)


def write_through_the_ladder(  # noqa: PLR0913 - one knob per contract under test
    root: Path,
    arm: str,
    step: int,
    *,
    fingerprint: str,
    replica_of: str | None,
    shift: float,
    coherence: CoherenceTriple = COHERENT,
    spans_absent: dict[str, dict[str, str]] | None = None,
) -> Path:
    """Write a cell with synthetic tensors through the ladder's own writer; return the cell directory."""
    activations = torch.arange(2 * TINY_LAYERS * TINY_HIDDEN, dtype=torch.float32).reshape(
        2, TINY_LAYERS, TINY_HIDDEN
    )
    cell_dir = step_dir(root, arm, step)
    write_full_weights_cell(
        cell_dir,
        hub_spec(arm, step, replica_of=replica_of),
        facts_with(fingerprint, root, label=f"{arm}@{step}"),
        identity=identity(),
        rows={"s": RowIndex(("a", "b"), ("A", "B"), ("p", "p"), (3, 3))},
        activations={("s", "mean"): activations + shift, ("s", "last"): activations - shift},
        spans_absent=spans_absent or {},
        run=RUN_FACTS,
        loading_report=LoadingReport(426, 1, 1),
        deltanet_kernel=FUSED_KERNELS,
        coherence=coherence,
        amplified=None,
        seconds=1.5,
        peak_vram_bytes=None,
    )
    return cell_dir


class TestWriterReaderContracts:
    def test_the_written_mirror_is_read_as_the_base_replica_and_displaces_exactly_zero(
        self, tmp_path: Path
    ) -> None:
        write_through_the_ladder(tmp_path, "base", 0, fingerprint="aa", replica_of=None, shift=0.0)
        write_through_the_ladder(
            tmp_path,
            "base-mirror",
            0,
            fingerprint="aa",
            replica_of="base:0",
            shift=0.0,
            coherence=CoherenceTriple.unavailable("a base mirror"),
            spans_absent={"s@head": {"b": "the rendering rewrote the whole check"}},
        )
        write_through_the_ladder(
            tmp_path, "tmax", 500, fingerprint="bb", replica_of=None, shift=0.5
        )
        ladder = load_full_weights_ladder(tmp_path)
        base, mirror, tmax = ladder.cells
        assert base.provenance[REPLICA_OF_FIELD] is None
        assert mirror.provenance[REPLICA_OF_FIELD] == "base:0"
        assert declared_replica_anchor(mirror) == base.label == "base/step-0"
        assert declared_replica_anchor(base) is None
        report = assert_replica_displacement_zero(base, mirror)
        assert report["max_abs_displacement"] == 0.0
        assert coherence_of(tmax) == COHERENT
        assert not coherence_of(mirror).available
        assert coherence_of(mirror).reason == "a base mirror"
        assert spans_absent_of(mirror) == {"s@head": {"b": "the rendering rewrote the whole check"}}
        assert spans_absent_of(base) == {}

    def test_a_mirror_declaring_another_cell_is_not_the_bases(self, tmp_path: Path) -> None:
        write_through_the_ladder(tmp_path, "base", 0, fingerprint="aa", replica_of=None, shift=0.0)
        write_through_the_ladder(
            tmp_path, "other", 0, fingerprint="cc", replica_of=None, shift=0.25
        )
        write_through_the_ladder(
            tmp_path, "mirror", 0, fingerprint="cc", replica_of="other:0", shift=0.25
        )
        ladder = load_full_weights_ladder(tmp_path)
        with pytest.raises(
            ReplicaDisplacementError, match="declares replica_of='other/step-0', not"
        ):
            assert_replica_displacement_zero(ladder.cell("base", 0), ladder.cell("mirror", 0))

    def test_a_manifest_with_the_provenance_key_renamed_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        write_through_the_ladder(tmp_path, "base", 0, fingerprint="aa", replica_of=None, shift=0.0)
        mirror_dir = write_through_the_ladder(
            tmp_path, "base-mirror", 0, fingerprint="aa", replica_of="base:0", shift=0.0
        )
        manifest_path = mirror_dir / CELL_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"]["replica-of"] = manifest["provenance"].pop(REPLICA_OF_FIELD)
        manifest_path.write_text(json.dumps(manifest))
        base = read_cell(step_dir(tmp_path, "base", 0))
        mirror = read_cell(mirror_dir)
        with pytest.raises(CellFormatError, match="records no 'replica_of' in its provenance"):
            assert_replica_displacement_zero(base, mirror)
        with pytest.raises(CellFormatError, match="records no 'replica_of' in its provenance"):
            load_full_weights_ladder(tmp_path)

    def test_the_coherence_block_round_trips_and_a_partial_one_is_refused(
        self, tmp_path: Path
    ) -> None:
        cell_dir = write_through_the_ladder(
            tmp_path, "tmax", 500, fingerprint="bb", replica_of=None, shift=0.5
        )
        cell = read_cell(cell_dir)
        assert cell.provenance[COHERENCE_FIELD] == COHERENT.to_payload()
        assert coherence_of(cell) == COHERENT
        manifest_path = cell_dir / CELL_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"][COHERENCE_FIELD]["truncation_share"] = None
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(CellFormatError, match="partly null"):
            coherence_of(read_cell(cell_dir))
        del manifest["provenance"][SPANS_ABSENT_FIELD]
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(CellFormatError, match="records no 'spans_absent'"):
            spans_absent_of(read_cell(cell_dir))


# --------------------------------------------------------------------------------------
# The sentence corpus: its render convention, and the set name the reader selects by
# --------------------------------------------------------------------------------------

SENTENCE_AXIS = 3


def sentence_cell(
    *, gap: float, seed: int, stimulus_set: str = CONCEPT_EVAL_AWARENESS
) -> CapturedCell:
    """Build a cell holding the eval-awareness rows the corpus writes, stored under `stimulus_set`, `e` planted."""
    stimuli = sentence_stimuli([CONCEPT_EVAL_AWARENESS])
    rows = row_index_for(stimuli, [20 for _ in stimuli])
    generator = torch.Generator().manual_seed(seed)
    n_pairs = len(stimuli) // 2
    per_pair = torch.randn(n_pairs, N_LAYERS, HIDDEN, generator=generator)
    matrix = per_pair.repeat_interleave(2, dim=0) + 0.3 * torch.randn(
        len(stimuli), N_LAYERS, HIDDEN, generator=generator
    )
    positive = [row for row, side in enumerate(rows.sides) if side == SENTENCE_SIDE_POSITIVE]
    matrix[positive, PLANTED_LAYER, :] += gap * axis(SENTENCE_AXIS)
    return cell_from(
        arm="base", rows={stimulus_set: rows}, activations={(stimulus_set, "mean"): matrix}
    )


class TestSentenceContract:
    def test_the_sentence_rows_template_here_and_refuse_verbatim(self) -> None:
        stimuli = sentence_stimuli([CONCEPT_EVAL_AWARENESS])
        rendered = render_stimuli(
            cast("Any", TinyTokenizer()),
            stimuli,
            convention=STIMULUS_RENDER_TEMPLATED,
            enable_thinking=True,
        )
        assert len(rendered) == len(stimuli)
        first = rendered[stimuli[0].stimulus_id]
        assert first.startswith(TURN_PREFIX)
        assert first.endswith(TURN_SUFFIX)
        assert first[len(TURN_PREFIX) : -len(TURN_SUFFIX)] == stimuli[0].text
        with pytest.raises(ValueError, match="do not begin with this template's user-turn prefix"):
            render_stimuli(
                cast("Any", TinyTokenizer()),
                stimuli,
                convention=STIMULUS_RENDER_VERBATIM,
                enable_thinking=True,
            )

    def test_e_is_fitted_from_the_set_under_its_written_name(self) -> None:
        reads = sentence_directions(
            sentence_cell(gap=4.0, seed=5),
            poolings=["mean"],
            layers=[0, PLANTED_LAYER],
            knobs=FAST,
            gate=UNIDENTIFIABLE,
        )
        assert {read.name for read in reads.reads} == {NAME_EVAL_AWARENESS}
        planted = read_at(reads.reads, layer=PLANTED_LAYER)
        assert planted.beats_placebo
        assert planted.n_pairs == len(SENTENCE_STIMULUS_SETS[CONCEPT_EVAL_AWARENESS])
        direction = reads.directions[NAME_EVAL_AWARENESS, "mean", STRATUM_ALL, PLANTED_LAYER]
        assert (
            abs(torch.nn.functional.cosine_similarity(direction, axis(SENTENCE_AXIS), dim=0)) > 0.9
        )
        assert not read_at(reads.reads, layer=0).beats_placebo

    def test_a_misspelled_set_is_refused_with_the_known_names_listed(self) -> None:
        cell = sentence_cell(gap=4.0, seed=5)
        with pytest.raises(
            DirectionRefusalError,
            match=r"'eval-awareness' is not a captured sentence set; the sets are "
            r"\['contradiction', 'deception', 'eval_awareness', 'shortcut'\]",
        ):
            sentence_directions(
                cell,
                poolings=["mean"],
                layers=[PLANTED_LAYER],
                knobs=FAST,
                gate=UNIDENTIFIABLE,
                stimulus_set="eval-awareness",
            )

    def test_a_cell_holding_the_set_under_another_name_is_refused_at_load(self) -> None:
        cell = sentence_cell(gap=4.0, seed=5, stimulus_set="eval-awareness")
        with pytest.raises(CellFormatError, match="holds no set 'eval_awareness'"):
            sentence_directions(
                cell, poolings=["mean"], layers=[PLANTED_LAYER], knobs=FAST, gate=UNIDENTIFIABLE
            )
