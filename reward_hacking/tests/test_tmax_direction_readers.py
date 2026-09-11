"""The TMAX direction fitters and readers on synthetic cells with planted structure, CPU only.

Every module under test is arithmetic over `games.interp_cells` cells, so the cells here are built in
memory with a known direction planted at a known layer and the reads have ground-truth answers: the
planted axis is recovered where it was planted and not elsewhere, the base against its mirror reads
exactly 0.0, row-permuted labels score near 0.5, a probe with too few positives is refused rather than
scored, sign-contaminated shuffled-label draws are excluded, and a decode at a far wrong layer fails the
stored-decode gate and returns no tokens. Lens reads run against a stub lens (one rotation per layer)
and a stub unembedding, which is enough to exercise every path without a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from games.eval_model import FullWeightsFacts, FullWeightsSource
from games.interp_cells import (
    CapturedCell,
    CellFormatError,
    CellIdentity,
    RowIndex,
    read_cell,
    step_dir,
    write_cell,
)
from reward_hacking.interp.lens_geometry import (
    GateProvenance,
    GateProvenanceError,
    StoredDecodeGate,
    cosine_with_band,
    rank_biased_overlap,
    standardize_direction,
    topk_overlap,
    wrong_layer_for,
)
from reward_hacking.interp.lens_pullback import affine_transport, lens_direction, pullback
from reward_hacking.interp.linear_probe import ProbeConfig
from reward_hacking.interp.tmax_capture_ladder import write_full_weights_cell
from reward_hacking.interp.tmax_directions import (
    CAPABILITY_CONTRAST,
    DISPOSITION_LABEL_IDENTIFIABLE,
    DISPOSITION_LABEL_UNIDENTIFIABLE,
    HACK_CONTRAST,
    NAME_TWIN,
    STRATA,
    STRATUM_ALL,
    STRATUM_DETECTABLE,
    STRATUM_NOT_DETECTABLE,
    DirectionRead,
    DirectionRefusal,
    DirectionSet,
    IdentifiabilityGate,
    ValidationKnobs,
    align_labels,
    generation_directions,
    load_directions,
    read_records_table,
    records_table,
    side_contrast_directions,
    twin_directions,
    within_group_pairs,
)
from reward_hacking.interp.tmax_displacement_series import (
    ReplicaDisplacementError,
    assert_replica_displacement_zero,
    displacement_directions,
    displacement_series,
    displacement_table,
)
from reward_hacking.interp.tmax_full_weights import (
    CaptureRunFacts,
    CoherenceTriple,
    FullWeightsCellSpec,
    LoadingReport,
)
from reward_hacking.interp.tmax_lens_reads import (
    MIN_DECODE_LAYER,
    DecodeContext,
    DecodeLayerError,
    DecodeRead,
    LensPair,
    LensReadBundle,
    TrackedPathError,
    decode_family,
    decode_with_floor,
    direction_cosines,
    lens_transfer_read,
    refuse_tracked_path,
    resolved_counts_from_summary,
)
from reward_hacking.interp.tmax_twin_sidecar import SIDE_HONEST, SIDE_RIGGED, SIDES, STIMULUS_SET
from reward_hacking.interp.transfer_matrix import (
    MIN_POSITIVE_GROUPS,
    MIN_POSITIVE_ROWS,
    ProbeSpec,
    TransferMatrix,
    transfer_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

HIDDEN = 32
N_LAYERS = 4
PLANTED_LAYER = 2
N_PROBLEMS = 24
GENERATION_SET = "tmax-generation"
FAST = ValidationKnobs(config=ProbeConfig(n_permutations=2), n_placebos=20, n_shuffled=5)
IDENTIFIABLE = IdentifiabilityGate(30, 9)
UNIDENTIFIABLE = IdentifiabilityGate(3, 2)

NARRATION_SUMMARY: dict[str, Any] = {
    "strata": [
        {
            "unit": "allenai/tmax-9b@step_500",
            "stratum": "all-strata",
            "noticed": 47,
            "complied": 54,
            "noticed_and_refused": 32,
        }
    ],
    "identifiability": {
        "min_rows": 25,
        "min_problems": 8,
        "stratum": "honest-pass-rigged",
        "definitions": {
            "outcome": {
                "definition": "outcome",
                "group": "noticed_outcome_honest",
                "stratum": "honest-pass-rigged",
                "rows": 59,
                "problems": 20,
                "identifiable": True,
                "by_unit": {
                    "allenai/tmax-9b@step_500": {"rows": 32, "problems": 14, "identifiable": True}
                },
            },
            "judge-disjoint": {
                "definition": "judge-disjoint",
                "group": "noticed_judge_not_complied",
                "stratum": "honest-pass-rigged",
                "rows": 35,
                "problems": 17,
                "identifiable": True,
                "by_unit": {
                    "allenai/tmax-9b@step_500": {"rows": 15, "problems": 9, "identifiable": False}
                },
            },
        },
    },
}
"""The Phase 1 Opus 5 readout's real counts in the summary's shape: the two definitions differ."""


def axis(index: int, dims: int = HIDDEN) -> torch.Tensor:
    direction = torch.zeros(dims)
    direction[index] = 1.0
    return direction


def identity_for(n_layers: int = N_LAYERS) -> CellIdentity:
    return CellIdentity(
        base_model="tiny/base",
        stimuli_sha256="digest",
        rendered_sha256="digest",
        layer_convention="post_block",
        n_layers=n_layers,
        hidden_size=HIDDEN,
        batch_size=1,
        compute_dtype="float32",
        store_dtype="float32",
        stimulus_render="verbatim",
    )


def cell_from(
    *,
    arm: str,
    rows: dict[str, RowIndex],
    activations: dict[tuple[str, str], torch.Tensor],
    provenance: dict[str, Any] | None = None,
) -> CapturedCell:
    return CapturedCell(
        arm=arm,
        step=0,
        identity=identity_for(),
        rows=rows,
        activations=activations,
        applied_adapter_weights=None,
        adapter_weights_sha256=None,
        provenance=provenance or {},
        source=Path(f"/nonexistent/{arm}"),
    )


# --------------------------------------------------------------------------------------
# Twin cells: six renderings per problem, the rigged side displaced along axis 0 at one layer
# --------------------------------------------------------------------------------------


def twin_rows(n_problems: int = N_PROBLEMS) -> RowIndex:
    return RowIndex(
        stimulus_ids=tuple(f"p{i}--{side}" for i in range(n_problems) for side in SIDES),
        sides=tuple(side for _ in range(n_problems) for side in SIDES),
        pair_ids=tuple(f"p{i}" for i in range(n_problems) for _ in SIDES),
        token_counts=tuple(40 for _ in range(n_problems * len(SIDES))),
    )


def twin_matrix(*, gap: float, seed: int, n_problems: int = N_PROBLEMS) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    per_problem = torch.randn(n_problems, N_LAYERS, HIDDEN, generator=generator)
    matrix = per_problem.repeat_interleave(len(SIDES), dim=0) + 0.3 * torch.randn(
        n_problems * len(SIDES), N_LAYERS, HIDDEN, generator=generator
    )
    rigged = [i * len(SIDES) + SIDES.index(SIDE_RIGGED) for i in range(n_problems)]
    matrix[rigged, PLANTED_LAYER, :] += gap * axis(0)
    return matrix


def detectable_flags(n_problems: int = N_PROBLEMS) -> dict[str, bool]:
    return {f"p{i}": i % 2 == 0 for i in range(n_problems)}


@pytest.fixture(scope="module")
def base_matrix() -> torch.Tensor:
    return twin_matrix(gap=4.0, seed=0)


@pytest.fixture(scope="module")
def base(base_matrix: torch.Tensor) -> CapturedCell:
    return cell_from(
        arm="base",
        rows={STIMULUS_SET: twin_rows()},
        activations={(STIMULUS_SET, "mean"): base_matrix},
    )


@pytest.fixture(scope="module")
def twin_reads(base: CapturedCell) -> DirectionSet:
    return twin_directions(
        base,
        detectable=detectable_flags(),
        poolings=["mean"],
        layers=list(range(N_LAYERS)),
        knobs=FAST,
        gate=IDENTIFIABLE,
    )


def read_at(reads: Sequence[DirectionRead], **where: object) -> DirectionRead:
    matches = [r for r in reads if all(getattr(r, k) == v for k, v in where.items())]
    assert len(matches) == 1, f"{len(matches)} reads match {where}"
    return matches[0]


class TestIdentifiabilityGate:
    def test_below_either_threshold_relabels(self) -> None:
        assert UNIDENTIFIABLE.disposition_label == DISPOSITION_LABEL_UNIDENTIFIABLE
        assert IdentifiabilityGate(25, 7).disposition_label == DISPOSITION_LABEL_UNIDENTIFIABLE
        assert IdentifiabilityGate(25, 8).disposition_label == DISPOSITION_LABEL_IDENTIFIABLE
        assert (
            IdentifiabilityGate.unlabelled().disposition_label == DISPOSITION_LABEL_UNIDENTIFIABLE
        )

    def test_reads_one_named_definition_out_of_the_judge_summary(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(NARRATION_SUMMARY))
        outcome = IdentifiabilityGate.from_narration_summary(path, definition="outcome")
        disjoint = IdentifiabilityGate.from_narration_summary(path, definition="judge-disjoint")
        assert (outcome.rows, outcome.problems, outcome.identifiable) == (59, 20, True)
        assert (disjoint.rows, disjoint.problems, disjoint.identifiable) == (35, 17, True)
        assert (outcome.definition, disjoint.definition) == ("outcome", "judge-disjoint")
        assert outcome.source == str(path)
        assert IdentifiabilityGate.unlabelled().definition == "unlabelled"

    def test_the_definition_is_required_and_refused_by_name_when_absent(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(NARRATION_SUMMARY))
        with pytest.raises(TypeError):
            IdentifiabilityGate.from_narration_summary(path)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="no identifiability definition 'majority'"):
            IdentifiabilityGate.from_narration_summary(path, definition="majority")
        flat = tmp_path / "flat.json"
        flat.write_text(
            json.dumps(
                {
                    "identifiability": {
                        "noticed_and_refused_rows": 59,
                        "noticed_and_refused_problems": 20,
                        "min_rows": 25,
                        "min_problems": 8,
                    }
                }
            )
        )
        with pytest.raises(ValueError, match="predates the outcome / judge-disjoint split"):
            IdentifiabilityGate.from_narration_summary(flat, definition="outcome")

    def test_resolved_counts_take_the_gate_group_under_the_named_definition(self) -> None:
        step = "allenai/tmax-9b@step_500"
        outcome = resolved_counts_from_summary(
            NARRATION_SUMMARY, unit=step, source="s", definition="outcome"
        )
        disjoint = resolved_counts_from_summary(
            NARRATION_SUMMARY, unit=step, source="s", definition="judge-disjoint"
        )
        assert (outcome.noticed, outcome.complied) == (47, 54)
        assert (outcome.noticed_and_refused, disjoint.noticed_and_refused) == (32, 15)
        assert disjoint.definition == "judge-disjoint"
        with pytest.raises(KeyError, match="no 'outcome' identifiability entry for unit"):
            resolved_counts_from_summary(
                NARRATION_SUMMARY, unit="nobody", source="s", definition="outcome"
            )


class TestTwinDirections:
    def test_the_planted_axis_is_recovered_where_it_was_planted(
        self, twin_reads: DirectionSet
    ) -> None:
        direction = twin_reads.directions[NAME_TWIN, "mean", STRATUM_ALL, PLANTED_LAYER]
        assert abs(torch.nn.functional.cosine_similarity(direction, axis(0), dim=0)) > 0.9
        read = read_at(twin_reads.reads, name=NAME_TWIN, stratum=STRATUM_ALL, layer=PLANTED_LAYER)
        assert read.beats_placebo
        assert read.clears_null
        assert read.beats_shuffled is True
        assert read.disposition_label == DISPOSITION_LABEL_IDENTIFIABLE

    def test_a_noise_layer_does_not_beat_its_placebo(self, twin_reads: DirectionSet) -> None:
        read = read_at(twin_reads.reads, name=NAME_TWIN, stratum=STRATUM_ALL, layer=0)
        assert not read.beats_placebo

    def test_strata_partition_the_problems(self, twin_reads: DirectionSet) -> None:
        by_stratum = {
            stratum: read_at(twin_reads.reads, name=NAME_TWIN, stratum=stratum, layer=PLANTED_LAYER)
            for stratum in STRATA
        }
        assert by_stratum[STRATUM_ALL].n_pairs == N_PROBLEMS
        assert by_stratum[STRATUM_DETECTABLE].n_pairs == N_PROBLEMS // 2
        assert by_stratum[STRATUM_NOT_DETECTABLE].n_pairs == N_PROBLEMS // 2
        assert by_stratum[STRATUM_DETECTABLE].beats_placebo

    def test_the_content_controls_share_no_planted_signal(self, twin_reads: DirectionSet) -> None:
        control = read_at(
            twin_reads.reads, name="d_comment_control", stratum=STRATUM_ALL, layer=PLANTED_LAYER
        )
        twin = read_at(twin_reads.reads, name=NAME_TWIN, stratum=STRATUM_ALL, layer=PLANTED_LAYER)
        assert twin.direction_accuracy > control.direction_accuracy

    def test_the_shuffled_cosine_gate_excludes_every_draw_at_zero_tolerance(
        self, base: CapturedCell
    ) -> None:
        strict = ValidationKnobs(
            config=FAST.config, n_placebos=5, n_shuffled=4, shuffled_max_abs_cosine=0.0
        )
        reads = side_contrast_directions(
            base,
            NAME_TWIN,
            positive_side=SIDE_RIGGED,
            negative_side=SIDE_HONEST,
            poolings=["mean"],
            layers=[PLANTED_LAYER],
            knobs=strict,
            gate=UNIDENTIFIABLE,
        )
        read = reads.reads[0]
        assert read.shuffled_n_excluded == 4
        assert len(read.shuffled_excluded_cosines) == 4
        assert read.shuffled_kept_accuracy_max is None
        assert read.beats_shuffled is None
        assert read.disposition_label == DISPOSITION_LABEL_UNIDENTIFIABLE

    def test_a_stratum_too_small_to_fold_is_a_refusal_row_not_a_hole(self) -> None:
        small = cell_from(
            arm="base",
            rows={STIMULUS_SET: twin_rows(6)},
            activations={(STIMULUS_SET, "mean"): twin_matrix(gap=4.0, seed=1, n_problems=6)},
        )
        reads = side_contrast_directions(
            small,
            NAME_TWIN,
            positive_side=SIDE_RIGGED,
            negative_side=SIDE_HONEST,
            poolings=["mean"],
            layers=[PLANTED_LAYER],
            knobs=FAST,
            gate=IDENTIFIABLE,
            detectable=detectable_flags(6),
            strata=STRATA,
        )
        assert {r.stratum for r in reads.reads} == {STRATUM_ALL}
        assert {r.stratum for r in reads.refusals} == {STRATUM_DETECTABLE, STRATUM_NOT_DETECTABLE}
        assert all("folds" in r.reason for r in reads.refusals)

    def test_tables_round_trip_through_ndjson_and_directions_reload(
        self, twin_reads: DirectionSet, tmp_path: Path
    ) -> None:
        twin_reads.save(tmp_path)
        reloaded = pl.read_ndjson(tmp_path / "direction-reads.ndjson", infer_schema_length=None)
        assert_frame_equal(reloaded, twin_reads.table(), check_dtypes=False)
        assert (
            read_records_table(tmp_path / "direction-refusals.ndjson", DirectionRefusal).height == 0
        )
        directions = load_directions(tmp_path / "directions.pt")
        assert directions.keys() == twin_reads.directions.keys()
        assert twin_reads.by_layer(NAME_TWIN, "mean").keys() == set(range(N_LAYERS))

    def test_the_planted_axis_survives_a_disk_round_trip(
        self, base_matrix: torch.Tensor, tmp_path: Path
    ) -> None:
        write_cell(
            step_dir(tmp_path, "base", 0),
            arm="base",
            step=0,
            identity=identity_for(),
            rows={STIMULUS_SET: twin_rows()},
            activations={(STIMULUS_SET, "mean"): base_matrix},
            applied_adapter_weights=None,
            adapter_weights_sha256=None,
            provenance={},
        )
        reads = side_contrast_directions(
            read_cell(step_dir(tmp_path, "base", 0)),
            NAME_TWIN,
            positive_side=SIDE_RIGGED,
            negative_side=SIDE_HONEST,
            poolings=["mean"],
            layers=[PLANTED_LAYER],
            knobs=FAST,
            gate=IDENTIFIABLE,
        )
        assert reads.reads[0].beats_placebo


# --------------------------------------------------------------------------------------
# Generation cells: (problem, unit) groups with item identity planted as a large group offset
# --------------------------------------------------------------------------------------

HACK_LAYER = 1
CAPABILITY_LAYER = 3
GROUP_SHAPE = (("hack", 2), ("honest-pass", 3), ("honest-fail", 2))


def generation_labels(n_problems: int, units: Sequence[str]) -> pl.DataFrame:
    rows = [
        {
            "stimulus_id": f"g{problem}-{unit}-{kind}-{sample}",
            "problem_id": f"p{problem}",
            "unit": unit,
            "hack": kind == "hack",
            "hidden_pass": kind == "honest-pass",
            "is_code": sample % 2 == 0,
        }
        for problem in range(n_problems)
        for unit in units
        for kind, count in GROUP_SHAPE
        for sample in range(count)
    ]
    return pl.DataFrame(rows)


def generation_cell(
    labels: pl.DataFrame, *, arm: str, seed: int, translation: float = 0.0
) -> CapturedCell:
    generator = torch.Generator().manual_seed(seed)
    n = labels.height
    matrix = 0.3 * torch.randn(n, N_LAYERS, HIDDEN, generator=generator)
    offsets: dict[tuple[str, str], torch.Tensor] = {}
    for row, record in enumerate(labels.iter_rows(named=True)):
        key = (str(record["problem_id"]), str(record["unit"]))
        if key not in offsets:
            offsets[key] = 3.0 * torch.randn(N_LAYERS, HIDDEN, generator=generator)
        matrix[row] += offsets[key]
        if record["hack"]:
            matrix[row, HACK_LAYER] += 2.0 * axis(5)
        if record["hidden_pass"]:
            matrix[row, CAPABILITY_LAYER] += 1.5 * axis(7)
        matrix[row, HACK_LAYER] += translation * axis(5)
    rows = RowIndex(
        stimulus_ids=tuple(labels["stimulus_id"].to_list()),
        sides=tuple("hack" if h else "honest" for h in labels["hack"].to_list()),
        pair_ids=tuple(labels["problem_id"].to_list()),
        token_counts=tuple(50 for _ in range(n)),
    )
    return cell_from(
        arm=arm, rows={GENERATION_SET: rows}, activations={(GENERATION_SET, "mean"): matrix}
    )


@pytest.fixture(scope="module")
def gen_labels() -> pl.DataFrame:
    return generation_labels(10, ["base", "step_500"])


@pytest.fixture(scope="module")
def gen_base(gen_labels: pl.DataFrame) -> CapturedCell:
    return generation_cell(gen_labels, arm="base", seed=3)


class TestGenerationDirections:
    def test_within_group_pairs_cycle_the_shorter_side_and_count_groups(
        self, gen_base: CapturedCell, gen_labels: pl.DataFrame
    ) -> None:
        aligned = align_labels(gen_base, GENERATION_SET, gen_labels)
        pairing = within_group_pairs(
            aligned,
            positive=HACK_CONTRAST.positive,
            negative=HACK_CONTRAST.negative,
            positive_side="hack",
            negative_side="honest-pass",
        )
        assert pairing.n_groups == 20
        assert pairing.layout.n_pairs == 40
        assert pairing.n_positive_rows == 40
        assert pairing.n_negative_rows == 60
        assert pairing.groups_without_both_sides == 0
        pair_ids = gen_labels["problem_id"].to_list()
        for pos, neg in zip(
            pairing.layout.positive_rows.tolist(),
            pairing.layout.negative_rows.tolist(),
            strict=True,
        ):
            assert pair_ids[pos] == pair_ids[neg]

    def test_d_hack_is_found_through_the_item_offsets(
        self, gen_base: CapturedCell, gen_labels: pl.DataFrame
    ) -> None:
        reads = generation_directions(
            gen_base,
            gen_labels,
            HACK_CONTRAST,
            stimulus_set=GENERATION_SET,
            poolings=["mean"],
            layers=[0, HACK_LAYER],
            knobs=FAST,
            gate=UNIDENTIFIABLE,
        )
        hack = read_at(reads.reads, layer=HACK_LAYER)
        direction = reads.directions["d_hack", "mean", STRATUM_ALL, HACK_LAYER]
        assert abs(torch.nn.functional.cosine_similarity(direction, axis(5), dim=0)) > 0.9
        assert hack.beats_placebo
        assert hack.n_positive_rows == 40
        assert hack.disposition_label == DISPOSITION_LABEL_UNIDENTIFIABLE
        assert not read_at(reads.reads, layer=0).beats_placebo

    def test_d_capability_lives_at_its_own_layer(
        self, gen_base: CapturedCell, gen_labels: pl.DataFrame
    ) -> None:
        reads = generation_directions(
            gen_base,
            gen_labels,
            CAPABILITY_CONTRAST,
            stimulus_set=GENERATION_SET,
            poolings=["mean"],
            layers=[CAPABILITY_LAYER],
            knobs=FAST,
            gate=IDENTIFIABLE,
        )
        direction = reads.directions["d_capability", "mean", STRATUM_ALL, CAPABILITY_LAYER]
        assert abs(torch.nn.functional.cosine_similarity(direction, axis(7), dim=0)) > 0.9

    def test_mismatched_labels_are_refused(
        self, gen_base: CapturedCell, gen_labels: pl.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="different rows"):
            align_labels(gen_base, GENERATION_SET, gen_labels.head(gen_labels.height - 1))


# --------------------------------------------------------------------------------------
# Displacement: the mirror sabotage, the planted shift, the series against a structureless floor
# --------------------------------------------------------------------------------------

SHIFT_AXIS = 9


def shifted_cell(base_matrix: torch.Tensor, *, arm: str, shift: float, seed: int) -> CapturedCell:
    generator = torch.Generator().manual_seed(seed)
    matrix = base_matrix.clone() + 0.05 * torch.randn(base_matrix.shape, generator=generator)
    matrix[:, PLANTED_LAYER, :] += shift * axis(SHIFT_AXIS)
    return cell_from(
        arm=arm, rows={STIMULUS_SET: twin_rows()}, activations={(STIMULUS_SET, "mean"): matrix}
    )


def structureless_floor(base_matrix: torch.Tensor, *, shift: float, seed: int) -> CapturedCell:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(base_matrix.shape, generator=generator)
    noise = noise / noise.norm(dim=-1, keepdim=True) * shift
    return cell_from(
        arm="permuted",
        rows={STIMULUS_SET: twin_rows()},
        activations={(STIMULUS_SET, "mean"): base_matrix + noise},
    )


def written_mirror(
    root: Path, matrix: torch.Tensor, *, replica_of: str | None, arm: str = "base-mirror"
) -> CapturedCell:
    """Write a mirror cell through the capture ladder's own writer and read it back off disk.

    The provenance block, `replica_of` included, is therefore exactly what the GPU driver writes: the
    CLI label of the anchor (`base:0`), not the games label the reader compares against.
    """
    spec = FullWeightsCellSpec(
        arm=arm,
        step=0,
        source=FullWeightsSource(repo_id="mirror/of-base", revision="main", local_dir=None),
        replica_of=replica_of,
    )
    facts = FullWeightsFacts(
        label="mirror",
        snapshot_dir=root,
        commit_sha=None,
        weights_sha256=(("model.safetensors", "aa"),),
        chat_template_sha256=None,
        declares_vision_config=True,
    )
    cell_dir = step_dir(root, arm, 0)
    write_full_weights_cell(
        cell_dir,
        spec,
        facts,
        identity=identity_for(),
        rows={STIMULUS_SET: twin_rows()},
        activations={(STIMULUS_SET, "mean"): matrix},
        spans_absent={},
        run=CaptureRunFacts(
            git_sha="deadbeef",
            torch_version="0.0",
            device="cpu",
            stimuli_file="stimuli.jsonl",
            spans_sidecar=None,
            span_names=(),
            tokenizer_id="tiny/base",
            tokenizer_revision=None,
        ),
        loading_report=LoadingReport(1, 0, 0),
        deltanet_kernel={"chunk_gated_delta_rule": "fused", "causal_conv1d_fn": "fused"},
        coherence=CoherenceTriple.unavailable("a base mirror"),
        amplified=None,
        seconds=0.0,
        peak_vram_bytes=None,
    )
    return read_cell(cell_dir)


class TestDisplacement:
    def test_base_against_its_declared_mirror_is_exactly_zero(
        self, base: CapturedCell, base_matrix: torch.Tensor, tmp_path: Path
    ) -> None:
        mirror = written_mirror(tmp_path, base_matrix.clone(), replica_of="base:0")
        assert mirror.provenance["replica_of"] == "base:0"
        report = assert_replica_displacement_zero(base, mirror)
        assert report["max_abs_displacement"] == 0.0
        assert report["mirror"] == "base-mirror/step-0"

    def test_a_mirror_off_by_one_value_is_red(
        self, base: CapturedCell, base_matrix: torch.Tensor, tmp_path: Path
    ) -> None:
        perturbed = base_matrix.clone()
        perturbed[3, 1, 4] += 1e-6
        mirror = written_mirror(tmp_path, perturbed, replica_of="base:0")
        with pytest.raises(ReplicaDisplacementError, match="displaced from its declared mirror"):
            assert_replica_displacement_zero(base, mirror)

    def test_an_undeclared_mirror_is_refused(
        self, base: CapturedCell, base_matrix: torch.Tensor, tmp_path: Path
    ) -> None:
        twin = written_mirror(tmp_path, base_matrix.clone(), replica_of=None)
        with pytest.raises(ReplicaDisplacementError, match="declares replica_of=None"):
            assert_replica_displacement_zero(base, twin)

    def test_a_mirror_of_some_other_cell_is_refused(
        self, base: CapturedCell, base_matrix: torch.Tensor, tmp_path: Path
    ) -> None:
        twin = written_mirror(tmp_path, base_matrix.clone(), replica_of="step_500:500")
        with pytest.raises(
            ReplicaDisplacementError,
            match="declares replica_of='step_500/step-500', not base/step-0",
        ):
            assert_replica_displacement_zero(base, twin)

    def test_a_cell_written_by_something_else_is_refused_by_field_name(
        self, base: CapturedCell, base_matrix: torch.Tensor
    ) -> None:
        hand_built = cell_from(
            arm="base-mirror",
            rows={STIMULUS_SET: twin_rows()},
            activations={(STIMULUS_SET, "mean"): base_matrix.clone()},
            provenance={"replica-of": "base:0"},
        )
        with pytest.raises(CellFormatError, match="records no 'replica_of'"):
            assert_replica_displacement_zero(base, hand_built)

    def test_the_planted_shift_is_recovered_and_the_floor_does_not_align(
        self, base: CapturedCell, base_matrix: torch.Tensor, tmp_path: Path
    ) -> None:
        step = shifted_cell(base_matrix, arm="step_500", shift=2.0, seed=11)
        floor = structureless_floor(base_matrix, shift=2.0, seed=12)
        reads, directions = displacement_directions(
            step,
            base,
            sides=[SIDE_RIGGED, SIDE_HONEST],
            poolings=["mean"],
            layers=list(range(N_LAYERS)),
            gate=IDENTIFIABLE,
            floor=floor,
            n_placebos=20,
        )
        planted = next(r for r in reads if r.side == SIDE_RIGGED and r.layer == PLANTED_LAYER)
        direction = directions[SIDE_RIGGED, "mean", PLANTED_LAYER]
        assert abs(torch.nn.functional.cosine_similarity(direction, axis(SHIFT_AXIS), dim=0)) > 0.95
        assert planted.split_half_cosine > 0.95
        assert planted.cosine_to_floor is not None
        assert abs(planted.cosine_to_floor) < 0.5
        assert planted.floor_norm is not None
        noise_layer = next(r for r in reads if r.side == SIDE_RIGGED and r.layer == 0)
        assert noise_layer.norm < planted.norm / 5
        table = displacement_table(reads)
        table.write_ndjson(tmp_path / "d.ndjson")
        assert_frame_equal(pl.read_ndjson(tmp_path / "d.ndjson"), table, check_dtypes=False)

    def test_the_series_grows_with_the_shift_and_the_floor_stays_at_the_placebo(
        self, base: CapturedCell, base_matrix: torch.Tensor
    ) -> None:
        checkpoints = {
            "step_200": shifted_cell(base_matrix, arm="step_200", shift=1.0, seed=21),
            "step_500": shifted_cell(base_matrix, arm="step_500", shift=2.0, seed=22),
        }
        series = displacement_series(
            base,
            checkpoints,
            sides=[SIDE_HONEST],
            poolings=["mean"],
            layers=[0, PLANTED_LAYER],
            gate=IDENTIFIABLE,
            floor=structureless_floor(base_matrix, shift=2.0, seed=23),
            n_placebos=20,
        )
        at = {(p.checkpoint, p.layer): p for p in series.points}
        early, late = at["step_200", PLANTED_LAYER], at["step_500", PLANTED_LAYER]
        assert late.heldout_projection > early.heldout_projection > 0.5
        assert late.heldout_projection == pytest.approx(2.0, abs=0.1)
        assert abs(late.heldout_projection_placebo) < 0.3
        assert late.floor_heldout_projection is not None
        assert late.floor_heldout_projection < 0.5 * late.heldout_projection
        assert late.split_half_cosine > 0.95
        assert late.n_fit_rows == N_PROBLEMS // 2
        assert late.n_projected_rows == N_PROBLEMS // 2
        cosines = series.axis_cosines(n_placebos=20, seed=0)
        planted = next(c for c in cosines if c.layer == PLANTED_LAYER)
        assert planted.cosine > 0.95 > planted.placebo_abs_cosine_max
        assert planted.attenuation_ceiling is not None


# --------------------------------------------------------------------------------------
# Transfer matrix: a pure translation shifts the honest margin and leaves the separation alone
# --------------------------------------------------------------------------------------

TRANSLATION = 1.0


@pytest.fixture(scope="module")
def matrix_cells(gen_labels: pl.DataFrame, gen_base: CapturedCell) -> dict[str, CapturedCell]:
    return {
        "base": gen_base,
        "step_500": generation_cell(gen_labels, arm="step_500", seed=3, translation=TRANSLATION),
    }


@pytest.fixture(scope="module")
def transfer(matrix_cells: dict[str, CapturedCell], gen_labels: pl.DataFrame) -> TransferMatrix:
    return transfer_matrix(
        matrix_cells,
        gen_labels,
        stimulus_set=GENERATION_SET,
        poolings=["mean"],
        layers=[HACK_LAYER],
        gate=IDENTIFIABLE,
        config=ProbeConfig(n_folds=5),
        n_bootstrap=50,
    )


class TestTransferMatrix:
    def test_the_hack_probe_transfers_as_a_translation(self, transfer: TransferMatrix) -> None:
        cells = {
            (c.source, c.target): c for c in transfer.cells if c.probe == "hack_vs_honest_pass"
        }
        diagonal = cells["base", "base"]
        assert diagonal.auc > 0.9
        assert diagonal.diagonal_auc_band_low <= diagonal.auc <= diagonal.diagonal_auc_band_high
        off = cells["base", "step_500"]
        assert off.auc > 0.9
        assert off.translation_shift > 0.5
        assert abs(off.separation_auc_shift) < 0.1
        assert not off.auc_outside_diagonal_band
        assert off.disposition_label == DISPOSITION_LABEL_IDENTIFIABLE

    def test_specificity_and_capability_probes_sit_beside_the_hack_probe(
        self, transfer: TransferMatrix
    ) -> None:
        roles = {c.probe: c.role for c in transfer.cells}
        assert roles["correct_vs_buggy"] == "capability"
        assert roles["code_vs_prose"] == "specificity"
        refused = {r.probe: r for r in transfer.refusals}
        assert "docstring_present" in refused
        assert "absent" in refused["docstring_present"].reason
        assert refused["docstring_present"].n_positive_rows is None

    def test_row_permuted_labels_score_near_chance(
        self, matrix_cells: dict[str, CapturedCell], gen_labels: pl.DataFrame
    ) -> None:
        permuted = transfer_matrix(
            matrix_cells,
            gen_labels,
            stimulus_set=GENERATION_SET,
            poolings=["mean"],
            layers=[HACK_LAYER],
            gate=IDENTIFIABLE,
            config=ProbeConfig(n_folds=5),
            n_bootstrap=20,
            permute_labels_seed=7,
        )
        hack = [c for c in permuted.cells if c.probe == "hack_vs_honest_pass"]
        assert hack
        assert all(c.labels_permuted for c in hack)
        assert all(0.3 < c.auc < 0.7 for c in hack), [c.auc for c in hack]

    @pytest.mark.parametrize(
        ("n_problems", "expected_rows", "expected_groups", "failing"),
        [(6, 12, 6, "groups"), (8, 16, 8, "rows")],
        ids=["too-few-groups", "too-few-rows"],
    )
    def test_a_thin_probe_is_a_refusal_row_with_its_counts_not_a_number(
        self, n_problems: int, expected_rows: int, expected_groups: int, failing: str
    ) -> None:
        thin_labels = generation_labels(n_problems, ["base"])
        thin = {"base": generation_cell(thin_labels, arm="base", seed=5)}
        result = transfer_matrix(
            thin,
            thin_labels,
            stimulus_set=GENERATION_SET,
            poolings=["mean"],
            layers=[HACK_LAYER],
            gate=IDENTIFIABLE,
            probes=[
                ProbeSpec("hack_vs_honest_pass", "hack", pl.col("hack"), ~pl.col("hack"), ("hack",))
            ],
            config=ProbeConfig(n_folds=5),
            n_bootstrap=5,
        )
        assert not result.cells
        (refusal,) = result.refusals
        assert (refusal.n_positive_rows, refusal.n_positive_groups) == (
            expected_rows,
            expected_groups,
        )
        assert (expected_rows < MIN_POSITIVE_ROWS, expected_groups < MIN_POSITIVE_GROUPS) == (
            True,
            failing == "groups",
        )
        assert refusal.reason == (
            f"{expected_rows} positive rows across {expected_groups} groups; a transfer number "
            f"needs at least {MIN_POSITIVE_ROWS} rows across {MIN_POSITIVE_GROUPS} groups"
        )

    def test_tables_round_trip_and_the_square_view_is_square(
        self, transfer: TransferMatrix, tmp_path: Path
    ) -> None:
        transfer.save(tmp_path)
        reloaded = pl.read_ndjson(tmp_path / "transfer-matrix.ndjson", infer_schema_length=None)
        assert_frame_equal(reloaded, transfer.table(), check_dtypes=False)
        square = transfer.square(
            probe="hack_vs_honest_pass", pooling="mean", layer=HACK_LAYER, metric="auc"
        )
        assert square.shape == (2, 3)
        assert square.columns == ["source", "base", "step_500"]


# --------------------------------------------------------------------------------------
# Lens reads against a stub lens: one rotation per layer, an identity unembedding
# --------------------------------------------------------------------------------------

LENS_DIM = 8
LENS_LAYERS = 32


class RotationLens:
    """Transport at layer L is a fixed random rotation R_L; a residual R_L^T e_k decodes to token k."""

    def __init__(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.rotations = [
            torch.linalg.qr(torch.randn(LENS_DIM, LENS_DIM, generator=generator))[0]
            for _ in range(LENS_LAYERS)
        ]

    def transport(self, direction: torch.Tensor, layer: int) -> torch.Tensor:
        return self.rotations[layer] @ direction


class IdentityUnembed:
    def unembed(self, transported: torch.Tensor) -> torch.Tensor:
        return transported


def token_name(token_id: int) -> str:
    return f"tok{token_id}"


def anchors_for(lens: RotationLens) -> StoredDecodeGate:
    targets = torch.arange(LENS_DIM)
    residuals = torch.stack(
        [
            torch.stack(
                [lens.rotations[layer].T @ axis(int(t), LENS_DIM) for layer in range(LENS_LAYERS)]
            )
            for t in targets
        ]
    )
    return StoredDecodeGate(residuals=residuals, target_token_ids=targets, lens_name="stub")


@pytest.fixture(scope="module")
def lens() -> RotationLens:
    return RotationLens(seed=0)


@pytest.fixture(scope="module")
def gate(lens: RotationLens) -> StoredDecodeGate:
    return anchors_for(lens)


def context_for(lens: RotationLens, gate: StoredDecodeGate | None) -> DecodeContext:
    return DecodeContext(
        lens=lens,
        model=IdentityUnembed(),
        lens_name="stub",
        id_to_token=token_name,
        top_k=3,
        n_placebos=5,
        gate=gate,
        disposition=IDENTIFIABLE,
        fitted_layers=frozenset(range(LENS_LAYERS)),
        gate_top_k=1,
    )


class TestLensGeometry:
    def test_overlaps(self) -> None:
        assert topk_overlap(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0
        assert topk_overlap(["a", "b", "c"], ["x", "y", "z"], 3) == 0.0
        assert rank_biased_overlap(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)
        assert rank_biased_overlap(["a", "b", "c"], ["c", "b", "a"]) < 1.0

    def test_standardization_keeps_the_norm_and_reweights(self) -> None:
        direction = torch.tensor([3.0, 4.0])
        scale = torch.tensor([3.0, 1.0])
        standardized = standardize_direction(direction, scale)
        assert float(standardized.norm()) == pytest.approx(5.0)
        assert standardized[1] > standardized[0]

    def test_cosine_band(self) -> None:
        banded = cosine_with_band(
            axis(0), axis(0), n_placebos=20, generator=torch.Generator().manual_seed(0)
        )
        assert banded.cosine == pytest.approx(1.0)
        assert banded.clears_band

    def test_wrong_layer_stays_in_range(self) -> None:
        assert wrong_layer_for(20, n_layers=32, offset=8) == 12
        assert wrong_layer_for(3, n_layers=32, offset=8) == 11
        with pytest.raises(ValueError, match="no layer"):
            wrong_layer_for(3, n_layers=6, offset=8)


class TestStoredDecodeGate:
    def test_the_right_layer_is_admitted_and_a_far_wrong_layer_loses(
        self, lens: RotationLens, gate: StoredDecodeGate
    ) -> None:
        result = gate.check(lens, IdentityUnembed(), 20, k=1)
        assert result.hit_rate == 1.0
        assert result.wrong_layer_hit_rate < 0.5
        assert result.admitted

    def test_anchors_mislabelled_by_eight_layers_fail_the_gate_and_the_decode_returns_no_tokens(
        self, lens: RotationLens, gate: StoredDecodeGate
    ) -> None:
        shifted = StoredDecodeGate(
            residuals=torch.roll(gate.residuals, shifts=8, dims=1),
            target_token_ids=gate.target_token_ids,
            lens_name="stub",
        )
        result = shifted.check(lens, IdentityUnembed(), 20, k=1)
        assert not result.admitted
        read = decode_with_floor(
            lens.rotations[20].T @ axis(2, LENS_DIM),
            context_for(lens, shifted),
            name="d_twin",
            layer=20,
        )
        assert read.tokens == ()
        assert read.gate_admitted is False
        assert read.placebo_overlap_mean is None

    def test_round_trip_carries_the_provenance_and_an_unstamped_gate_is_not_written(
        self, gate: StoredDecodeGate, tmp_path: Path
    ) -> None:
        with pytest.raises(GateProvenanceError, match="no provenance"):
            gate.save(tmp_path)
        stamped = StoredDecodeGate(
            residuals=gate.residuals,
            target_token_ids=gate.target_token_ids,
            lens_name="stub",
            provenance=GateProvenance(
                lens_name="stub",
                model_label="stub@0",
                weights_identity="hf:0",
                weights_fingerprint="f" * 64,
                lens_sha256="c" * 64,
                anchors_path="anchors.jsonl",
                anchors_sha256="d" * 64,
                anchor_digest="e" * 64,
                anchor_rows=tuple((f"anchor:{i:03d}", 16) for i in range(gate.n_anchors)),
                skip_first=16,
                jlens_commit="581d398",
                load_path="stub",
            ),
        )
        stamped.save(tmp_path)
        loaded = StoredDecodeGate.load(tmp_path)
        assert torch.equal(loaded.residuals, gate.residuals)
        assert loaded.lens_name == "stub"
        assert loaded.provenance == stamped.provenance


class TestLensReads:
    def test_a_direction_decodes_to_its_token_above_the_placebo_floor(
        self, lens: RotationLens, gate: StoredDecodeGate
    ) -> None:
        read = decode_with_floor(
            lens.rotations[20].T @ axis(2, LENS_DIM),
            context_for(lens, gate),
            name="d_twin",
            layer=20,
        )
        assert read.tokens[0] == "tok2"
        assert read.gate_admitted is True
        assert read.placebo_overlap_max is not None
        assert read.placebo_overlap_max < 1.0
        assert read.variant == "raw"

    def test_layers_below_the_floor_are_refused_or_skipped(
        self, lens: RotationLens, gate: StoredDecodeGate
    ) -> None:
        context = context_for(lens, gate)
        with pytest.raises(DecodeLayerError):
            decode_with_floor(axis(0, LENS_DIM), context, name="d_twin", layer=MIN_DECODE_LAYER - 1)
        reads = decode_family({8: axis(0, LENS_DIM), 20: axis(1, LENS_DIM)}, context, name="d_twin")
        assert [r.layer for r in reads] == [20]

    def test_the_standardized_variant_is_emitted_where_a_scale_exists(
        self, lens: RotationLens
    ) -> None:
        context = DecodeContext(
            lens=lens,
            model=IdentityUnembed(),
            lens_name="stub",
            id_to_token=token_name,
            top_k=3,
            n_placebos=2,
            gate=None,
            disposition=IDENTIFIABLE,
            fitted_layers=frozenset(range(LENS_LAYERS)),
            scale_by_layer={20: torch.ones(LENS_DIM)},
        )
        reads = decode_family(
            {20: axis(1, LENS_DIM), 21: axis(1, LENS_DIM)}, context, name="d_disp"
        )
        assert [(r.layer, r.variant) for r in reads] == [
            (20, "raw"),
            (20, "standardized"),
            (21, "raw"),
        ]

    def test_lens_transfer_reads_one_against_itself_and_a_wrong_layer(
        self, lens: RotationLens
    ) -> None:
        pair = LensPair(name="base", lens=lens, model=IdentityUnembed())
        direction = lens.rotations[20].T @ (axis(2, LENS_DIM) + 0.5 * axis(3, LENS_DIM))
        read = lens_transfer_read(
            direction,
            name="d_twin",
            layer=20,
            reference=pair,
            other=pair,
            id_to_token=token_name,
            top_k=3,
            n_placebos=5,
            wrong_layer=12,
            disposition=IDENTIFIABLE,
            halves=(direction, direction),
        )
        assert read.overlap == 1.0
        assert read.placebo_overlap_mean == 1.0
        assert read.wrong_layer_overlap < 1.0
        assert read.split_half_overlap == 1.0

    def test_direction_cosines_carry_roles_and_bands(self) -> None:
        rows = direction_cosines(
            {20: axis(0)},
            reference_name="e",
            targets={"d_twin": {20: axis(0), 21: axis(1)}},
            controls={"d_comment_control": {20: axis(1)}},
            n_placebos=10,
        )
        assert [(r.second, r.role, r.layer) for r in rows] == [
            ("d_twin", "target", 20),
            ("d_comment_control", "content-control", 20),
        ]
        assert rows[0].clears_band
        assert not rows[1].clears_band

    def test_the_bundle_refuses_a_tracked_path_and_writes_an_untracked_one(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        (repo / "reward_hacking").mkdir(parents=True)
        with pytest.raises(TrackedPathError):
            refuse_tracked_path(repo / "reward_hacking" / "out", repo_root=repo)
        assert (
            refuse_tracked_path(repo / "artifacts" / "run", repo_root=repo)
            == (repo / "artifacts" / "run").resolve()
        )
        assert (
            refuse_tracked_path(tmp_path / "elsewhere", repo_root=repo)
            == (tmp_path / "elsewhere").resolve()
        )
        bundle = LensReadBundle()
        with pytest.raises(TrackedPathError):
            bundle.save(repo / "docs" / "notes", repo_root=repo)
        written = bundle.save(tmp_path / "prefix", repo_root=repo)
        assert (written / "lens-decodes.ndjson").exists()
        assert read_records_table(written / "lens-decodes.ndjson", DecodeRead).height == 0


class TestLensPullback:
    def test_the_affine_transport_is_recovered_exactly(self, lens: RotationLens) -> None:
        jacobian, bias = affine_transport(lens, 20, dim=LENS_DIM)
        assert torch.allclose(jacobian, lens.rotations[20], atol=1e-6)
        assert torch.allclose(bias, torch.zeros_like(bias))

    def test_the_pullback_transports_onto_its_contrast_and_placebos_do_not(
        self, lens: RotationLens
    ) -> None:
        contrast = axis(2, LENS_DIM) - axis(5, LENS_DIM)
        pulled, read = lens_direction(
            lens,
            20,
            contrast,
            dim=LENS_DIM,
            n_placebos=20,
            generator=torch.Generator().manual_seed(0),
        )
        assert torch.allclose(pulled, pullback(lens.rotations[20], contrast))
        assert read.alignment == pytest.approx(1.0)
        assert read.beats_placebo


class TestRecordsTable:
    def test_none_valued_columns_keep_their_type_and_round_trip(self, tmp_path: Path) -> None:
        read = DirectionRead(
            name="x",
            pooling="mean",
            stratum="all",
            layer=1,
            n_pairs=5,
            n_positive_rows=5,
            n_negative_rows=5,
            probe_accuracy=0.5,
            probe_null_accuracy_max=0.5,
            clears_null=False,
            direction_accuracy=0.5,
            placebo_accuracy_mean=0.5,
            placebo_accuracy_max=0.5,
            accuracy_empirical_p=0.5,
            beats_placebo=False,
            split_half_cosine=0.0,
            direction_norm=1.0,
            shuffled_n_draws=0,
            shuffled_n_excluded=0,
            shuffled_excluded_cosines=(),
            shuffled_kept_accuracy_max=None,
            beats_shuffled=None,
            disposition_label="d",
        )
        table = records_table([read], DirectionRead)
        assert table.schema["shuffled_kept_accuracy_max"] == pl.Float64
        assert table.schema["beats_shuffled"] == pl.Boolean
        table.write_ndjson(tmp_path / "t.ndjson")
        assert_frame_equal(pl.read_ndjson(tmp_path / "t.ndjson", schema=table.schema), table)
