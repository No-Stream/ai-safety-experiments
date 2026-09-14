"""Run the displacement analysis on planted geometry, where every answer is known in advance.

Offline and CPU-only over cells written straight to disk, so the fixture ladder is the real path
rather than a stand-in for it. The plant is chosen so that each claim the module exists to separate
has a different right answer: one arm shifts the whole population *along* a planted contrast axis, a
second shifts it by the same amount *against* that axis, and a third shifts only the pairs of one
`cited_cells` stratum. A pooled cross-arm cosine cannot tell those three apart; the reads here have
to.

What each class pins:

:class:`TestPlantedDisplacement` -- the planted displacement is recovered where it was planted and
nowhere else, the two opposite arms read cosine -1 against a split-half ceiling of ~1 and a placebo
floor near 1/sqrt(hidden), and the axis split puts the whole movement in the signed along-axis
component with an empty residual. This is the shape the real question is asked in: "moved
orthogonally" versus "moved oppositely along one shared axis".

:class:`TestStratifiedReads` -- the strata actually slice. The arm that only moves its
`matched-column` pairs reads full displacement inside that stratum, noise inside the other, and the
row-weighted average pooled -- and the selections that would merely repeat their pooled parent are
skipped *and* recorded rather than silently dropped.

:class:`TestNulls` -- the two nulls, each watched to fail. Base-against-base is exactly zero, and
stays a real check only because a deliberately row-misaligned displacement makes it raise. The
shuffled-label null lands at the placebo floor while the real axis clears it.

:class:`TestTwoPathAgreement` -- the check from plan step 1d, plus the three ways it has to behave
badly on purpose: a layer-shifted second path collapses the agreement, a second path whose layers
are all identical makes the off-by-one arm undetectable and therefore raises, and a post-final-norm
top layer is reported apart from the comparable band instead of counting as disagreement.

:class:`TestReportArtifacts` -- the payload shape a readout will consume, and the corpus-identity
refusal reached through the CLI.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
import torch

from games import interp_displacement
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    CellIdentity,
    Stimulus,
    load_ladder,
    load_stimuli,
    row_index_for,
    step_dir,
    stimuli_digest,
    write_cell,
)
from games.interp_displacement import (
    AGREEMENT_FILENAME,
    FRAMING_NOT_APPLICABLE,
    POOLED_SET_GROUP,
    REPORT_FILENAME,
    AgreementCheckError,
    PipelineNullError,
    RowGroup,
    _axis_component,
    _resolve_axis_references,
    base_vs_base_null,
    build_parser,
    build_row_groups,
    displacement_vectors,
    load_framing_strata,
    run,
    two_path_agreement,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

N_LAYERS = 3
PLANTED_LAYER = 1
HIDDEN = 128
POOLING = "last"

# The base cell's own geometry: a side-A-versus-side-B separation along dimension 0, in noise. The
# width, pair count, separation and noise are not free parameters -- a matched-norm placebo retains
# 1/sqrt(hidden) of a planted separation, so the real axis only clears its placebo while
# `SEPARATION << NOISE_STD * sqrt(HIDDEN)`, and a diff-of-means over `n_pairs` pairs only points
# where it was planted while `SEPARATION >> NOISE_STD * sqrt(2 * HIDDEN / n_pairs)`. At 64
# dimensions with separation 3.0 and noise 0.1 -- the first thing tried here -- a random direction
# separated the sides as perfectly as the real axis did, and the shuffled-label null had no floor to
# land at.
SEPARATION = 2.4
NOISE_STD = 0.3

# The planted population shift, reached at MAX_STEP and half of it at the intermediate checkpoint,
# plus a small arm-specific per-row jitter so an unplanted layer's displacement is small rather than
# exactly zero (a zero displacement has no direction, and every cosine against it would be nan).
DISPLACEMENT = 1.0
ARM_JITTER = 0.02
MAX_STEP = 20
STEPS = (10, 20)

COMMITMENT_SET = "commitment-contrast"
TRANSFER_SET = "decision-transfer"
N_COMMITMENT_PAIRS = 12
N_TRANSFER_PAIRS = 4
MATCHED = "matched-column"
SPLIT = "split-diagonal-offdiagonal"
SCENARIO = "decision-scenario"
CORRELATED = "correlated-instance"
RECORDED = "recorded-decision"
N_MATCHED_PAIRS = 4

ALONG_ARM = "shifted-along-axis"
AGAINST_ARM = "shifted-against-axis"
STRATUM_ARM = "shifted-in-matched-column-only"

N_PLACEBOS = 20


def planted_axis() -> torch.Tensor:
    """The one unit direction the fixture separates and displaces along."""
    axis = torch.zeros(HIDDEN)
    axis[0] = 1.0
    return axis


def pair_index_of(pair_id: str) -> int:
    """The integer index encoded in a fixture pair id."""
    return int(pair_id.rsplit("p", 1)[1])


def cited_stratum_of(stimulus: Stimulus) -> str:
    """The planted citation regime: the transfer set has none, the first pairs are matched-column."""
    if stimulus.stimulus_set == TRANSFER_SET:
        return SCENARIO
    return MATCHED if pair_index_of(stimulus.pair_id) < N_MATCHED_PAIRS else SPLIT


def framing_half_of(stimulus: Stimulus) -> str:
    """The planted counterpart framing: alternating halves on the game set, none on the transfer set."""
    if stimulus.stimulus_set == TRANSFER_SET:
        return FRAMING_NOT_APPLICABLE
    return CORRELATED if pair_index_of(stimulus.pair_id) % 2 == 0 else RECORDED


def make_set(stimulus_set: str, n_pairs: int) -> list[Stimulus]:
    """Two-sided stimuli for one set, in the A, B, A, B order the corpus writer emits."""
    return [
        Stimulus(
            stimulus_id=f"{stimulus_set}--p{index}--{side}",
            stimulus_set=stimulus_set,
            side=side,
            pair_id=f"{stimulus_set}--p{index}",
            text=f"rendered text for {stimulus_set} pair {index} side {side}",
        )
        for index in range(n_pairs)
        for side in ("A", "B")
    ]


def make_stimuli() -> list[Stimulus]:
    """The whole fixture corpus: one game-shaped set and one transfer set."""
    return [
        *make_set(COMMITMENT_SET, N_COMMITMENT_PAIRS),
        *make_set(TRANSFER_SET, N_TRANSFER_PAIRS),
    ]


def provenance_row(stimulus: Stimulus) -> dict[str, Any]:
    """One provenance record, in the two schemas the real files carry."""
    row: dict[str, Any] = {
        "id": stimulus.stimulus_id,
        "set": stimulus.stimulus_set,
        "side": stimulus.side,
        "pair_id": stimulus.pair_id,
        "stance": "planted",
    }
    if stimulus.stimulus_set == TRANSFER_SET:
        row["scenario_id"] = stimulus.pair_id
        return row
    row["cited_cells"] = cited_stratum_of(stimulus)
    row["counterpart_framing"] = framing_half_of(stimulus)
    return row


def write_corpus(
    tmp_path: Path,
    stimuli: list[Stimulus],
    *,
    mutate_provenance: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> tuple[Path, Path]:
    """Write the stimulus corpus and its provenance file, optionally mutating provenance rows."""
    stimuli_path = tmp_path / "stimuli.jsonl"
    stimuli_path.write_text(
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
    rows = [provenance_row(stimulus) for stimulus in stimuli]
    if mutate_provenance is not None:
        rows = mutate_provenance(rows)
    provenance_path = tmp_path / "stimuli_provenance.jsonl"
    provenance_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return stimuli_path, provenance_path


def identity_for(digest: str, *, store_dtype: str) -> CellIdentity:
    """The identity every cell of one fixture ladder shares."""
    return CellIdentity(
        base_model="Qwen/Qwen3.5-2B",
        stimuli_sha256=digest,
        rendered_sha256=f"rendered-{digest}",
        layer_convention="post_block",
        n_layers=N_LAYERS,
        hidden_size=HIDDEN,
        batch_size=1,
        compute_dtype="bfloat16",
        store_dtype=store_dtype,
        stimulus_render="verbatim",
    )


def displacement_of(stimulus: Stimulus, arm: str, step: int) -> float:
    """The planted along-axis shift for one row: which arm, which checkpoint, which stratum."""
    scale = DISPLACEMENT * step / MAX_STEP
    if arm == ALONG_ARM:
        return scale
    if arm == AGAINST_ARM:
        return -scale
    return scale if cited_stratum_of(stimulus) == MATCHED else 0.0


def plant_activations(stimuli: list[Stimulus], *, arm: str, step: int) -> torch.Tensor:
    """One cell's activations: the base geometry, plus this arm's planted shift and jitter."""
    axis = planted_axis()
    base_generator = torch.Generator().manual_seed(11)
    matrix = torch.randn(len(stimuli), N_LAYERS, HIDDEN, generator=base_generator) * NOISE_STD
    for row, stimulus in enumerate(stimuli):
        sign = 1.0 if stimulus.side == "A" else -1.0
        matrix[row, PLANTED_LAYER, :] += sign * SEPARATION / 2.0 * axis
    if arm == BASE_ARM:
        return matrix
    jitter_generator = torch.Generator().manual_seed(abs(hash(arm)) % 10_000 + step)
    matrix += torch.randn(matrix.shape, generator=jitter_generator) * ARM_JITTER
    for row, stimulus in enumerate(stimuli):
        matrix[row, PLANTED_LAYER, :] += displacement_of(stimulus, arm, step) * axis
    return matrix


def write_planted_cell(  # noqa: PLR0913 - a planted cell is a root, a corpus, an arm, a step, a dtype
    root: Path,
    stimuli: list[Stimulus],
    digest: str,
    *,
    arm: str,
    step: int,
    store_dtype: str = "float32",
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Path:
    """Write one cell of a fixture ladder, optionally passing its tensors through `transform`."""
    by_set: dict[str, list[Stimulus]] = {}
    for stimulus in stimuli:
        by_set.setdefault(stimulus.stimulus_set, []).append(stimulus)
    matrix = plant_activations(stimuli, arm=arm, step=step)
    if transform is not None:
        matrix = transform(matrix)
    rows: dict[str, Any] = {}
    activations: dict[tuple[str, str], torch.Tensor] = {}
    offset = 0
    for name, members in by_set.items():
        rows[name] = row_index_for(members, [64] * len(members))
        activations[name, POOLING] = matrix[offset : offset + len(members)].clone()
        offset += len(members)
    return write_cell(
        step_dir(root, arm, step),
        arm=arm,
        step=step,
        identity=identity_for(digest, store_dtype=store_dtype),
        rows=rows,
        activations=activations,
        applied_adapter_weights=None if arm == BASE_ARM else 372,
        adapter_weights_sha256=None if arm == BASE_ARM else f"{arm}-{step}",
        provenance={"git_sha": "testing"},
    ).parent


def plant_ladder(
    root: Path,
    stimuli: list[Stimulus],
    digest: str,
    *,
    store_dtype: str = "float32",
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Path:
    """Write the base anchor plus the three planted arms at both checkpoints."""
    write_planted_cell(
        root,
        stimuli,
        digest,
        arm=BASE_ARM,
        step=BASE_STEP,
        store_dtype=store_dtype,
        transform=transform,
    )
    for arm in (ALONG_ARM, AGAINST_ARM, STRATUM_ARM):
        for step in STEPS:
            write_planted_cell(
                root,
                stimuli,
                digest,
                arm=arm,
                step=step,
                store_dtype=store_dtype,
                transform=transform,
            )
    return root


def read_of(
    payload: dict[str, Any],
    *,
    step: int = MAX_STEP,
    layer: int = PLANTED_LAYER,
    set_group: str = POOLED_SET_GROUP,
    stratum: str = "pooled",
) -> dict[str, Any]:
    """One read out of the payload, by the coordinates the assertions care about."""
    matches = [
        read
        for read in payload["reads"]
        if read["step"] == step
        and read["layer"] == layer
        and read["set_group"] == set_group
        and read["stratum"] == stratum
    ]
    assert len(matches) == 1, f"expected one read, got {len(matches)}"
    return matches[0]


def arm_of(read: dict[str, Any], arm: str) -> dict[str, Any]:
    """One arm's entry inside a read."""
    return next(item for item in read["arms"] if item["arm"] == arm)


def pair_of(read: dict[str, Any], arm_a: str, arm_b: str) -> dict[str, Any]:
    """One arm pair's entry inside a read, in whichever order the module emitted it."""
    return next(
        item for item in read["arm_pairs"] if {item["arm_a"], item["arm_b"]} == {arm_a, arm_b}
    )


def axis_of(read: dict[str, Any], target: str, *, axis_set: str, reference: str) -> dict[str, Any]:
    """One axis-component entry inside a read."""
    return next(
        item
        for item in read["axis_components"]
        if item["target"] == target
        and item["axis_set"] == axis_set
        and item["axis_reference"] == reference
    )


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Plant one ladder, run the whole CLI over it once, and share the result.

    Module-scoped because reading the ladder is the expensive part and every assertion below is a
    different question about the same numbers.
    """
    root = tmp_path_factory.mktemp("displacement")
    stimuli = make_stimuli()
    digest = stimuli_digest(stimuli)
    capture_root = plant_ladder(root / "capture", stimuli, digest)
    frozen_root = plant_ladder(
        root / "frozen",
        stimuli,
        digest,
        store_dtype="float16",
    )
    stimuli_path, provenance_path = write_corpus(root, stimuli)
    out_dir = root / "analysis"
    args = build_parser().parse_args(
        [
            "--capture-root",
            str(capture_root),
            "--stimuli",
            str(stimuli_path),
            "--provenance",
            str(provenance_path),
            "--out-dir",
            str(out_dir),
            "--positive-side",
            "A",
            "--n-placebos",
            str(N_PLACEBOS),
            "--null-layer",
            str(PLANTED_LAYER),
            "--frozen-cells",
            str(frozen_root),
        ]
    )
    payload = run(args)
    return {
        "payload": payload,
        "root": root,
        "capture_root": capture_root,
        "frozen_root": frozen_root,
        "stimuli": stimuli,
        "stimuli_path": stimuli_path,
        "provenance_path": provenance_path,
        "out_dir": out_dir,
    }


class TestPlantedDisplacement:
    def test_the_planted_shift_is_recovered_at_the_planted_layer(self, planted: Any) -> None:
        read = read_of(planted["payload"])
        along = arm_of(read, ALONG_ARM)
        assert along["displacement_norm"] == pytest.approx(DISPLACEMENT, abs=0.05)
        elsewhere = arm_of(read_of(planted["payload"], layer=0), ALONG_ARM)
        assert elsewhere["displacement_norm"] < DISPLACEMENT / 5

    def test_the_shift_grows_with_the_checkpoint(self, planted: Any) -> None:
        early = arm_of(read_of(planted["payload"], step=10), ALONG_ARM)
        late = arm_of(read_of(planted["payload"], step=20), ALONG_ARM)
        assert early["displacement_norm"] == pytest.approx(DISPLACEMENT / 2, abs=0.05)
        assert late["displacement_norm"] > early["displacement_norm"]

    def test_both_residual_denominators_are_reported(self, planted: Any) -> None:
        """A displacement norm is unreadable without the scale of the states it moved."""
        read = read_of(planted["payload"])
        along = arm_of(read, ALONG_ARM)
        assert along["relative_to_residual_mean_vector"] == pytest.approx(
            along["displacement_norm"] / read["residual_mean_vector_norm"]
        )
        assert along["relative_to_residual_row_norm_mean"] == pytest.approx(
            along["displacement_norm"] / read["residual_row_norm_mean"]
        )
        assert read["residual_row_norm_mean"] > read["residual_mean_vector_norm"]

    def test_the_opposite_arms_read_minus_one_between_floor_and_ceiling(self, planted: Any) -> None:
        read = read_of(planted["payload"])
        pair = pair_of(read, ALONG_ARM, AGAINST_ARM)
        assert pair["cosine_real"] == pytest.approx(-1.0, abs=0.01)
        assert pair["same_axis_ceiling_pairs"] == pytest.approx(1.0, abs=0.02)
        assert read["placebo_abs_cosine_max"] < 0.6
        assert read["placebo_abs_cosine_mean"] < 0.3

    def test_the_cross_half_cosines_reproduce_the_cross_arm_read(self, planted: Any) -> None:
        """Disjoint halves cannot share stimulus-sampling noise, so this is the sharper version."""
        pair = pair_of(read_of(planted["payload"]), ALONG_ARM, AGAINST_ARM)
        for value in pair["cosine_cross_halves_pairs"]:
            assert value == pytest.approx(-1.0, abs=0.05)

    def test_the_row_split_is_flagged_as_a_side_split(self, planted: Any) -> None:
        """The corpus stores pairs A, B, A, B, so a row-parity split is every A against every B."""
        read = read_of(planted["payload"])
        assert read["row_parity_equals_side"] is True
        assert arm_of(read, ALONG_ARM)["split_half_cosine_rows"] == pytest.approx(1.0, abs=0.02)

    def test_the_axis_split_is_exact_arithmetic(self) -> None:
        """The split itself, where the answer has no estimation error: 3 along, 4 across, cosine 0.6."""
        vector = torch.zeros(HIDDEN)
        vector[0], vector[1] = 3.0, 4.0
        axis = torch.zeros(HIDDEN)
        axis[0] = 2.0
        read = _axis_component(
            "planted", vector, axis, axis_set=COMMITMENT_SET, axis_reference="base"
        )
        assert read.projection == pytest.approx(3.0)
        assert read.residual_norm == pytest.approx(4.0)
        assert read.cosine_to_axis == pytest.approx(0.6)
        assert read.axis_norm == pytest.approx(2.0)

    def test_the_movement_lands_in_the_along_axis_component(self, planted: Any) -> None:
        """The claim a bare cosine cannot make: how much of the shift is along the contrast axis.

        Attenuated against the planted value, and one-sidedly so: the axis it projects onto is a
        diff-of-means over 12 pairs, not the planted direction, and a finite-sample axis can only
        lose signal. So the bound is a band -- the same attenuation applies to the real captures.
        """
        read = read_of(planted["payload"])
        along = axis_of(read, ALONG_ARM, axis_set=COMMITMENT_SET, reference="base")
        against = axis_of(read, AGAINST_ARM, axis_set=COMMITMENT_SET, reference="base")
        assert 0.8 * DISPLACEMENT < along["projection"] <= 1.05 * DISPLACEMENT
        assert -1.05 * DISPLACEMENT <= against["projection"] < -0.8 * DISPLACEMENT
        assert along["residual_norm"] < along["projection"]
        assert along["cosine_to_axis"] > 0.8

    def test_an_unplanted_layer_carries_no_along_axis_component(self, planted: Any) -> None:
        """The other half of "lands where it was planted": nothing along the axis one layer down."""
        along = axis_of(
            read_of(planted["payload"], layer=0),
            ALONG_ARM,
            axis_set=COMMITMENT_SET,
            reference="base",
        )
        assert abs(along["cosine_to_axis"]) < 0.5
        assert along["residual_norm"] > abs(along["projection"])

    def test_the_difference_vector_is_split_along_the_same_axis(self, planted: Any) -> None:
        read = read_of(planted["payload"])
        difference = axis_of(
            read, f"{AGAINST_ARM}-minus-{ALONG_ARM}", axis_set=COMMITMENT_SET, reference="base"
        )
        along = axis_of(read, ALONG_ARM, axis_set=COMMITMENT_SET, reference="base")
        against = axis_of(read, AGAINST_ARM, axis_set=COMMITMENT_SET, reference="base")
        assert difference["projection"] == pytest.approx(
            against["projection"] - along["projection"], abs=1e-4
        )
        assert difference["projection"] < -1.6 * DISPLACEMENT

    def test_the_arms_own_axis_is_reported_beside_the_anchors(self, planted: Any) -> None:
        """The plan asks for the cell's own axis; the scratch read used the anchor's. Both, labelled."""
        read = read_of(planted["payload"])
        own = axis_of(read, ALONG_ARM, axis_set=COMMITMENT_SET, reference="own")
        base = axis_of(read, ALONG_ARM, axis_set=COMMITMENT_SET, reference="base")
        assert own["projection"] == pytest.approx(base["projection"], abs=0.05)

    def test_a_layer_by_step_matrix_carries_the_shape(self, planted: Any) -> None:
        matrix = next(
            item
            for item in planted["payload"]["matrices"]
            if item["metric"] == "displacement_relative_norm"
            and item["arm"] == ALONG_ARM
            and item["set_group"] == POOLED_SET_GROUP
            and item["stratum"] == "pooled"
        )
        assert matrix["steps"] == list(STEPS)
        assert matrix["layers"] == list(range(N_LAYERS))
        planted_column = [row[PLANTED_LAYER] for row in matrix["values"]]
        assert planted_column[1] > planted_column[0]
        assert min(planted_column) > max(row[0] for row in matrix["values"])


class TestStratifiedReads:
    def test_the_stratum_arm_moves_only_inside_its_stratum(self, planted: Any) -> None:
        payload = planted["payload"]
        matched = arm_of(read_of(payload, stratum=MATCHED), STRATUM_ARM)
        split = arm_of(read_of(payload, stratum=SPLIT), STRATUM_ARM)
        pooled = arm_of(read_of(payload), STRATUM_ARM)
        assert matched["displacement_norm"] == pytest.approx(DISPLACEMENT, abs=0.05)
        assert split["displacement_norm"] < DISPLACEMENT / 5
        assert (
            split["displacement_norm"] < pooled["displacement_norm"] < matched["displacement_norm"]
        )

    def test_the_pooled_read_is_the_row_weighted_average(self, planted: Any) -> None:
        """8 of 32 rows carry the shift, so the pooled displacement is a quarter of it."""
        pooled = arm_of(read_of(planted["payload"]), STRATUM_ARM)
        n_rows = read_of(planted["payload"])["n_rows"]
        matched_rows = read_of(planted["payload"], stratum=MATCHED)["n_rows"]
        assert pooled["displacement_norm"] == pytest.approx(
            DISPLACEMENT * matched_rows / n_rows, abs=0.05
        )

    def test_both_stratifications_are_read(self, planted: Any) -> None:
        payload = planted["payload"]
        strata = {
            (read["stratification"], read["stratum"])
            for read in payload["reads"]
            if read["set_group"] == POOLED_SET_GROUP
        }
        assert strata == {
            ("pooled", "pooled"),
            ("cited_cells", MATCHED),
            ("cited_cells", SPLIT),
            ("cited_cells", SCENARIO),
            ("counterpart_framing", CORRELATED),
            ("counterpart_framing", RECORDED),
            ("counterpart_framing", FRAMING_NOT_APPLICABLE),
        }

    def test_the_framing_halves_split_the_game_set_evenly(self, planted: Any) -> None:
        correlated = read_of(planted["payload"], stratum=CORRELATED)
        recorded = read_of(planted["payload"], stratum=RECORDED)
        assert correlated["n_rows"] == recorded["n_rows"] == N_COMMITMENT_PAIRS
        assert correlated["n_pairs"] == N_COMMITMENT_PAIRS // 2

    def test_a_redundant_selection_is_skipped_and_recorded(self, planted: Any) -> None:
        """The transfer set is one stratum, so slicing it would repeat its pooled read."""
        skipped = planted["payload"]["skipped_selections"]
        repeated = [
            item
            for item in skipped
            if item["set_group"] == TRANSFER_SET and item["stratum"] == SCENARIO
        ]
        assert len(repeated) == 1
        assert "repeats the pooled read" in repeated[0]["reason"]
        empty = [item for item in skipped if item["n_rows"] == 0]
        assert empty
        assert all("no rows" in item["reason"] for item in empty)

    def test_every_kept_selection_holds_whole_pairs(self, planted: Any) -> None:
        for read in planted["payload"]["reads"]:
            assert read["n_rows"] == 2 * read["n_pairs"]
            assert read["row_parity_equals_side"] is True


class TestNulls:
    def test_base_against_base_is_exactly_zero(self, planted: Any) -> None:
        null = planted["payload"]["base_vs_base_null"]
        assert null["max_abs_displacement"] == 0.0
        assert null["n_checks"] > 0

    def test_a_row_misaligned_displacement_makes_the_null_fail(
        self, planted: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sabotage: difference the base cell against a rolled copy of its own rows.

        This is the bug the null exists to catch -- two sides of a subtraction that are not the same
        rows -- and without watching it go red the zero above is just a reassuring message.
        """
        ladder = load_ladder(planted["capture_root"])
        base = ladder.cell(BASE_ARM, BASE_STEP)
        groups, _ = build_row_groups(
            base,
            stimulus_sets=[COMMITMENT_SET, TRANSFER_SET],
            stratifications={},
        )

        def misaligned(
            arm_rows: dict[str, torch.Tensor], base_rows: dict[str, torch.Tensor], group: RowGroup
        ) -> Any:
            rolled = {name: torch.roll(rows, shifts=1, dims=0) for name, rows in base_rows.items()}
            return displacement_vectors(arm_rows, rolled, group)

        monkeypatch.setattr(interp_displacement, "displacement_vectors", misaligned)
        with pytest.raises(PipelineNullError, match="displaced from itself"):
            base_vs_base_null(base, groups=groups, poolings=[POOLING])

    def test_the_shuffled_label_axis_lands_at_the_floor(self, planted: Any) -> None:
        null = next(
            item
            for item in planted["payload"]["shuffled_label_nulls"]["reads"]
            if item["stimulus_set"] == COMMITMENT_SET
        )
        assert null["real_direction_accuracy"] > null["real_placebo_accuracy_max"]
        assert null["clears_placebo"] is False
        assert abs(null["cosine_null_to_real"]) < 0.9
        assert 0 < null["n_flipped"] < null["n_pairs"]

    def test_a_set_too_small_to_hold_out_says_so_rather_than_vanishing(self, planted: Any) -> None:
        """Four transfer pairs cannot fill five folds, and a missing null must not read as a clean one."""
        skipped = planted["payload"]["shuffled_label_nulls"]["skipped"]
        assert [item["stimulus_set"] for item in skipped] == [TRANSFER_SET]
        assert skipped[0]["n_pairs"] == N_TRANSFER_PAIRS
        assert "cannot fill" in skipped[0]["reason"]


class TestTwoPathAgreement:
    def read_ladders(
        self, planted: Any, *, transform: Any = None, root_name: str = "variant"
    ) -> Any:
        """The canonical ladder plus a second path, optionally passed through `transform`."""
        canonical = load_ladder(planted["capture_root"])
        if transform is None:
            return canonical, load_ladder(planted["frozen_root"])
        variant_root = planted["root"] / root_name
        if not variant_root.exists():
            plant_ladder(
                variant_root,
                planted["stimuli"],
                stimuli_digest(planted["stimuli"]),
                store_dtype="float16",
                transform=transform,
            )
        return canonical, load_ladder(variant_root)

    def agreement(self, canonical: Any, frozen: Any, **overrides: Any) -> dict[str, Any]:
        """Run the agreement check over two ladders with the fixture's usual knobs."""
        from games.interp_cells import pair_layout  # noqa: PLC0415 - one local use, in a helper

        layouts = {
            name: pair_layout(canonical.cells[0], name, positive_side="A")
            for name in canonical.stimulus_sets
        }
        kwargs: dict[str, Any] = {
            "layouts": layouts,
            "poolings": [POOLING],
            "layers": list(range(N_LAYERS)),
            "agreement_floor": 0.99,
            "top_layer_post_norm": False,
        }
        kwargs.update(overrides)
        return two_path_agreement(canonical, frozen, **kwargs)

    def test_two_faithful_paths_agree_and_the_off_by_one_arm_goes_red(self, planted: Any) -> None:
        summary = planted["payload"]["two_path_agreement"]
        assert summary["direction_cosine_min"] > 0.99
        assert summary["n_below_floor"] == 0
        assert summary["off_by_one_sabotage"]["is_red"] is True
        assert summary["off_by_one_sabotage"]["cosine_max"] < 0.99
        assert summary["off_by_one_sabotage"]["n_reads"] > 0

    def test_a_path_one_layer_off_is_refused_by_name(self, planted: Any) -> None:
        """The plan's sabotage row, and what it does when the *paths* are the thing that is shifted.

        Rolling the second path down by one makes the off-by-one arm the correctly aligned
        comparison, so it agrees. The module cannot then certify anything and says which of the two
        readings applies, rather than reporting a summary whose same-layer disagreement it has no way
        to interpret.
        """
        canonical, shifted = self.read_ladders(
            planted,
            transform=lambda matrix: torch.roll(matrix, shifts=1, dims=1),
            root_name="rolled-one",
        )
        with pytest.raises(AgreementCheckError, match="ALREADY ONE LAYER APART"):
            self.agreement(canonical, shifted)

    def test_a_misaligned_path_shows_up_as_disagreement(self, planted: Any) -> None:
        """Two layers off: the same-layer comparison collapses and the off-by-one arm stays red."""
        canonical, shifted = self.read_ladders(
            planted,
            transform=lambda matrix: torch.roll(matrix, shifts=2, dims=1),
            root_name="rolled-two",
        )
        summary = self.agreement(canonical, shifted)["summary"]
        assert summary["n_below_floor"] > 0
        assert summary["direction_cosine_min"] < 0.99
        assert summary["off_by_one_sabotage"]["is_red"] is True

    def test_a_path_with_no_depth_structure_makes_the_check_refuse(self, planted: Any) -> None:
        """If the off-by-one arm cannot go red, the green reading means nothing, so this raises."""
        canonical, flat = self.read_ladders(
            planted,
            transform=lambda matrix: matrix[:, :1, :].expand(-1, N_LAYERS, -1).clone(),
            root_name="flat",
        )
        with pytest.raises(AgreementCheckError, match="off-by-one comparison agreed"):
            self.agreement(canonical, flat)

    def test_a_post_norm_top_layer_is_reported_apart_not_as_disagreement(
        self, planted: Any
    ) -> None:
        """The fifth format difference the plan's table missed, on the real frozen capture."""
        scale = 1.0 + torch.arange(2 * (N_COMMITMENT_PAIRS + N_TRANSFER_PAIRS), dtype=torch.float32)

        def rescale_top_layer(matrix: torch.Tensor) -> torch.Tensor:
            scaled = matrix.clone()
            scaled[:, N_LAYERS - 1, :] *= scale.unsqueeze(1)
            return scaled

        canonical, post_norm = self.read_ladders(
            planted, transform=rescale_top_layer, root_name="post-norm-top"
        )
        counted = self.agreement(canonical, post_norm)["summary"]
        assert counted["n_below_floor"] > 0

        apart = self.agreement(canonical, post_norm, top_layer_post_norm=True)["summary"]
        assert apart["n_below_floor"] == 0
        assert apart["direction_cosine_min"] > 0.99
        assert apart["post_norm_layer_reads"] > 0
        assert apart["post_norm_direction_cosine_min"] < 0.99


class TestReportArtifacts:
    def test_the_cli_writes_both_reports_with_their_context(self, planted: Any) -> None:
        out_dir = planted["out_dir"]
        assert (out_dir / REPORT_FILENAME).is_file()
        assert (out_dir / AGREEMENT_FILENAME).is_file()
        on_disk = json.loads((out_dir / REPORT_FILENAME).read_text())
        context = on_disk["context"]
        assert context["stimuli_sha256"] == stimuli_digest(planted["stimuli"])
        assert context["arms"] == sorted([ALONG_ARM, AGAINST_ARM, STRATUM_ARM])
        assert context["axis_references"] == ["base", "own"]
        assert context["null_layer"] == PLANTED_LAYER
        agreement = json.loads((out_dir / AGREEMENT_FILENAME).read_text())
        assert agreement["summary"] == on_disk["two_path_agreement"]
        assert agreement["reads"]

    def test_every_read_carries_its_denominators(self, planted: Any) -> None:
        """A displacement without its row count and its floor is not a readable number."""
        for read in planted["payload"]["reads"]:
            assert read["n_rows"] > 0
            assert read["placebo_abs_cosine_max"] > 0
            assert read["residual_row_norm_mean"] > 0
            assert len(read["arms"]) == 3
            assert len(read["arm_pairs"]) == 3

    def test_a_mismatched_corpus_stops_the_analysis(self, tmp_path: Path) -> None:
        """The identity guard, reached through the CLI: edited corpus, same ids, refused."""
        stimuli = make_stimuli()
        digest = stimuli_digest(stimuli)
        capture_root = plant_ladder(tmp_path / "capture", stimuli, digest)
        edited = [*stimuli[:-1], replace(stimuli[-1], text="an edited rendering")]
        stimuli_path, provenance_path = write_corpus(tmp_path, edited)
        args = build_parser().parse_args(
            [
                "--capture-root",
                str(capture_root),
                "--stimuli",
                str(stimuli_path),
                "--provenance",
                str(provenance_path),
                "--out-dir",
                str(tmp_path / "analysis"),
                "--positive-side",
                "A",
                "--n-placebos",
                "2",
            ]
        )
        with pytest.raises(ValueError, match="different stimulus corpus"):
            run(args)

    def test_a_pair_straddling_framing_halves_is_refused(self, tmp_path: Path) -> None:
        stimuli = make_stimuli()

        def straddle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            rows[1]["counterpart_framing"] = RECORDED
            rows[0]["counterpart_framing"] = CORRELATED
            return rows

        stimuli_path, provenance_path = write_corpus(tmp_path, stimuli, mutate_provenance=straddle)
        with pytest.raises(ValueError, match="cannot be sliced as one"):
            load_framing_strata(provenance_path, load_stimuli(stimuli_path))

    def test_an_unknown_axis_reference_is_named(self) -> None:
        with pytest.raises(ValueError, match="axis-references"):
            _resolve_axis_references("base,rotated")
