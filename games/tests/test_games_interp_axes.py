"""Pin the cited-cells conditioning: stratified axis geometry, and the refusals that keep it honest.

Offline and CPU-only, over synthetic cells with planted directions. The planted construction is the
confound itself in miniature: two stimulus sets share a direction inside one citation stratum while
one set's *other* stratum carries an orthogonal direction -- so the pooled axis-to-axis cosine is
genuinely between, the within-stratum cosine is genuinely high, and the within-axis stratum
contrast is genuinely near zero. A module that failed to condition would collapse those three
numbers into one.

The refusal tests are the other half: a provenance row that cannot be stratified, a provenance file
from a different rendering, a pair straddling two strata, and a corpus edited after capture must
all refuse loudly, because each would otherwise stratify tensors by the wrong text and produce a
plausible number.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from games.interp_axes import (
    KNOWN_SURFACE_RESIDUALS,
    POOLED_STRATUM,
    SCENARIO_STRATUM,
    ProvenanceError,
    assert_no_pair_crosses_split,
    build_construct_splits,
    build_parser,
    load_strata,
    run,
)
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    CellFormatError,
    CellIdentity,
    Stimulus,
    load_stimuli,
    row_index_for,
    step_dir,
    stimuli_digest,
    write_cell,
)

N_LAYERS = 3
HIDDEN = 64
SEPARATION = 3.0
NOISE_STD = 0.1

COARSE_SET = "axis-coarse"
LEAD_SET = "axis-lead"
DECISION_SET = "axis-decision"
MATCHED = "matched-column"
SPLIT = "split-diagonal-offdiagonal"


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


def make_construct_set(construct: str, n_pairs: int = 4) -> list[Stimulus]:
    """Small explicit grouped construct corpus used only for split validation."""
    rows: list[Stimulus] = []
    for pair_index in range(n_pairs):
        split = "fit" if pair_index % 2 == 0 else "heldout"
        group = f"{construct}-scenario-{pair_index}"
        rows.extend(
            Stimulus(
                stimulus_id=f"{construct}--p{pair_index}--{side}",
                stimulus_set=construct,
                side=side,
                pair_id=f"{construct}--p{pair_index}",
                text=f"fixed mechanics premise {construct} pair {pair_index} side {side}",
                metadata={
                    "construct": construct,
                    "scenario_group": group,
                    "split": split,
                    "measurement_boundary": "pre_action",
                    "action_commitment_present": False,
                    **(
                        {
                            "procedure_regime": (
                                "stochastic" if pair_index in (0, 1) else "deterministic"
                            ),
                            "dependence_mechanism": (
                                "shared-randomness"
                                if pair_index in (0, 1) and side == "A"
                                else "independent-randomness"
                                if pair_index in (0, 1)
                                else "shared-deterministic-procedure"
                                if side == "A"
                                else "independent-deterministic-procedure"
                            ),
                        }
                        if construct == "decision-dependence"
                        else {}
                    ),
                },
            )
            for side in ("A", "B")
        )
    return rows


def coarse_stratum_of(pair_index: int) -> str:
    """The planted stratum split for the coarse set: first half matched, second half split."""
    return MATCHED if pair_index < 4 else SPLIT


def provenance_row(stimulus: Stimulus) -> dict[str, Any]:
    """One provenance record matching the real files' two schemas."""
    row: dict[str, Any] = {
        "id": stimulus.stimulus_id,
        "set": stimulus.stimulus_set,
        "side": stimulus.side,
        "pair_id": stimulus.pair_id,
        "stance": "planted",
    }
    if stimulus.stimulus_set == DECISION_SET:
        row["scenario_id"] = stimulus.pair_id
        row["option_order"] = "canonical"
        return row
    pair_index = int(stimulus.pair_id.rsplit("p", 1)[1])
    row["cited_cells"] = (
        coarse_stratum_of(pair_index) if stimulus.stimulus_set == COARSE_SET else SPLIT
    )
    return row


class TestConstructSplits:
    def test_authored_group_split_is_used_and_pairs_stay_whole(self) -> None:
        stimuli = make_construct_set("costly-other-regard") + make_construct_set(
            "decision-dependence"
        )
        splits = build_construct_splits(stimuli, pairs_per_construct=4)
        for construct, split in splits.items():
            assert split.fit_groups == tuple(f"{construct}-scenario-{index}" for index in (0, 2))
            assert split.heldout_groups == tuple(
                f"{construct}-scenario-{index}" for index in (1, 3)
            )
            assert set(split.fit_pair_ids) | set(split.heldout_pair_ids) == {
                f"{construct}--p{index}" for index in range(4)
            }
            if construct == "decision-dependence":
                assert set(split.procedure_regime_by_pair.values()) == {
                    "stochastic",
                    "deterministic",
                }
            assert_no_pair_crosses_split(
                [stimulus for stimulus in stimuli if stimulus.stimulus_set == construct],
                split.fit_pair_ids,
                split.heldout_pair_ids,
            )

    def test_group_crossing_and_reserved_identity_are_refused(self) -> None:
        stimuli = make_construct_set("costly-other-regard") + make_construct_set(
            "decision-dependence"
        )
        crossed = [
            replace(
                stimulus,
                metadata={
                    **stimulus.metadata,
                    "scenario_group": "costly-other-regard-scenario-0",
                },
            )
            if stimulus.stimulus_id == "costly-other-regard--p1--B"
            else stimulus
            for stimulus in stimuli
        ]
        with pytest.raises(ValueError, match="straddles scenario/template groups"):
            build_construct_splits(crossed, pairs_per_construct=4)
        with pytest.raises(ValueError, match="reserved training/evaluation group"):
            build_construct_splits(
                stimuli,
                pairs_per_construct=4,
                reserved_groups=("decision-dependence-scenario-1",),
            )
        with pytest.raises(ValueError, match="reserved training/evaluation identity"):
            build_construct_splits(
                stimuli,
                pairs_per_construct=4,
                external_pair_ids=("decision-dependence--p1",),
            )

    def test_missing_or_mixed_explicit_split_is_refused(self) -> None:
        stimuli = make_construct_set("costly-other-regard") + make_construct_set(
            "decision-dependence"
        )
        missing = [
            replace(
                stimulus,
                metadata={key: value for key, value in stimulus.metadata.items() if key != "split"},
            )
            if stimulus.stimulus_id == "costly-other-regard--p0--A"
            else stimulus
            for stimulus in stimuli
        ]
        with pytest.raises(ValueError, match="must declare one shared metadata split"):
            build_construct_splits(missing, pairs_per_construct=4)

    def test_strict_decision_control_metadata_is_required_for_cooperation_path(self) -> None:
        stimuli = make_construct_set("costly-other-regard") + make_construct_set(
            "decision-dependence"
        )
        missing = [
            replace(
                stimulus,
                metadata={
                    key: value
                    for key, value in stimulus.metadata.items()
                    if key not in {"procedure_regime", "dependence_mechanism"}
                },
            )
            if stimulus.stimulus_set == "decision-dependence"
            else stimulus
            for stimulus in stimuli
        ]
        with pytest.raises(ValueError, match="dependence_mechanism"):
            build_construct_splits(
                missing,
                pairs_per_construct=4,
                require_decision_control=True,
            )


def write_corpus(
    tmp_path: Path, *, mutate_provenance: Any | None = None
) -> tuple[Path, Path, list[Stimulus]]:
    """Write the stimulus corpus and its provenance file, optionally mutating provenance rows."""
    stimuli = [*make_set(COARSE_SET, 8), *make_set(LEAD_SET, 8), *make_set(DECISION_SET, 4)]
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
    return stimuli_path, provenance_path, stimuli


def planted_direction(stimulus: Stimulus) -> torch.Tensor:
    """Which unit direction a stimulus's set and stratum separate along.

    The lead set and the coarse set's SPLIT stratum share dimension 0 -- the miniature of the shared
    citation regime -- while the coarse MATCHED stratum separates along the orthogonal dimension 1.
    """
    direction = torch.zeros(HIDDEN)
    pair_index = int(stimulus.pair_id.rsplit("p", 1)[1])
    if stimulus.stimulus_set == COARSE_SET and coarse_stratum_of(pair_index) == MATCHED:
        direction[1] = 1.0
    else:
        direction[0] = 1.0
    return direction


def write_planted_cell(root: Path, stimuli: list[Stimulus], digest: str) -> Path:
    """Write one base cell whose activations carry the planted per-stratum separations."""
    identity = CellIdentity(
        base_model="Qwen/Qwen3.5-2B",
        stimuli_sha256=digest,
        rendered_sha256=f"rendered-{digest}",
        layer_convention="post_block",
        n_layers=N_LAYERS,
        hidden_size=HIDDEN,
        batch_size=1,
        compute_dtype="bfloat16",
        store_dtype="float32",
        stimulus_render="verbatim",
    )
    generator = torch.Generator().manual_seed(7)
    rows: dict[str, Any] = {}
    activations: dict[tuple[str, str], torch.Tensor] = {}
    by_set: dict[str, list[Stimulus]] = {}
    for stimulus in stimuli:
        by_set.setdefault(stimulus.stimulus_set, []).append(stimulus)
    for stimulus_set, members in by_set.items():
        matrix = torch.zeros(len(members), N_LAYERS, HIDDEN)
        for row, stimulus in enumerate(members):
            sign = 1.0 if stimulus.side == "A" else -1.0
            base = sign * SEPARATION * planted_direction(stimulus)
            noise = torch.randn(N_LAYERS, HIDDEN, generator=generator) * NOISE_STD
            matrix[row] = base + noise
        rows[stimulus_set] = row_index_for(members, [11] * len(members))
        activations[stimulus_set, "mean"] = matrix
    return write_cell(
        step_dir(root, BASE_ARM, BASE_STEP),
        arm=BASE_ARM,
        step=BASE_STEP,
        identity=identity,
        rows=rows,
        activations=activations,
        applied_adapter_weights=None,
        adapter_weights_sha256=None,
        provenance={"git_sha": "testing"},
    ).parent


def run_axes(tmp_path: Path, *, n_folds: int = 2, n_placebos: int = 20) -> dict[str, Any]:
    """Build the corpus and one planted cell, run the CLI path end to end, return the payload."""
    stimuli_path, provenance_path, stimuli = write_corpus(tmp_path)
    root = tmp_path / "capture"
    write_planted_cell(root, stimuli, stimuli_digest(stimuli))
    args = build_parser().parse_args(
        [
            "--capture-root",
            str(root),
            "--stimuli",
            str(stimuli_path),
            "--provenance",
            str(provenance_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--positive-side",
            "A",
            "--n-folds",
            str(n_folds),
            "--n-placebos",
            str(n_placebos),
        ]
    )
    return run(args)


def cell_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The single base cell's reads."""
    return payload["cells"][f"{BASE_ARM}/step-{BASE_STEP}"]


def pair_reads(
    payload: dict[str, Any], set_a: str, set_b: str, stratum: str
) -> list[dict[str, Any]]:
    """Axis-pair reads for one (set_a, set_b, stratum), every layer."""
    return [
        read
        for read in cell_payload(payload)["axis_pairs"]
        if (read["set_a"], read["set_b"], read["stratum"]) == (set_a, set_b, stratum)
    ]


class TestStratifiedCosines:
    def test_within_stratum_cosine_separates_the_confound(self, tmp_path: Path) -> None:
        """Pooled sits between the high within-stratum cosine and the placebo floor."""
        payload = run_axes(tmp_path)
        within = pair_reads(payload, COARSE_SET, LEAD_SET, SPLIT)
        pooled = pair_reads(payload, COARSE_SET, LEAD_SET, POOLED_STRATUM)
        assert len(within) == N_LAYERS
        assert len(pooled) == N_LAYERS
        for read in within:
            assert read["cosine_real"] > 0.9
            assert read["cosine_real"] > read["placebo_abs_cosine_max"]
        for read in pooled:
            assert 0.5 < read["cosine_real"] < 0.9

    def test_stratum_contrast_reads_near_zero_when_strata_differ(self, tmp_path: Path) -> None:
        """The coarse set's two citation regimes were planted orthogonal; the contrast says so."""
        payload = run_axes(tmp_path)
        contrasts = [
            read
            for read in cell_payload(payload)["stratum_contrasts"]
            if read["stimulus_set"] == COARSE_SET
        ]
        assert len(contrasts) == N_LAYERS
        for read in contrasts:
            assert (read["stratum_a"], read["stratum_b"]) == (MATCHED, SPLIT)
            assert abs(read["cosine_real"]) < 0.3

    def test_quality_reads_clear_their_placebo_on_the_planted_axis(self, tmp_path: Path) -> None:
        payload = run_axes(tmp_path)
        pooled_lead = [
            read
            for read in cell_payload(payload)["quality"]
            if read["stimulus_set"] == LEAD_SET and read["stratum"] == POOLED_STRATUM
        ]
        assert len(pooled_lead) == N_LAYERS
        for read in pooled_lead:
            assert read["direction_accuracy"] == 1.0
            assert read["placebo_accuracy_mean"] < 1.0
            assert read["split_half_cosine"] > 0.9
            assert read["unavailable_reason"] is None

    def test_scenario_rows_form_their_own_stratum(self, tmp_path: Path) -> None:
        payload = run_axes(tmp_path)
        decision_strata = {
            read["stratum"]
            for read in cell_payload(payload)["quality"]
            if read["stimulus_set"] == DECISION_SET
        }
        assert decision_strata == {POOLED_STRATUM, SCENARIO_STRATUM}

    def test_the_residual_note_rides_every_payload(self, tmp_path: Path) -> None:
        payload = run_axes(tmp_path)
        assert payload["known_surface_residuals"] == list(KNOWN_SURFACE_RESIDUALS)

    def test_the_primary_stratum_is_recorded(self, tmp_path: Path) -> None:
        """The confound-clean regime is the primary comparison, and the payload says which it is."""
        payload = run_axes(tmp_path)
        assert payload["context"]["primary_stratum"] == "matched-column"

    def test_directions_are_saved_per_stratum(self, tmp_path: Path) -> None:
        run_axes(tmp_path)
        out = tmp_path / "out" / "directions" / BASE_ARM / f"step-{BASE_STEP}"
        names = sorted(path.name for path in out.iterdir())
        assert f"{COARSE_SET}--{MATCHED}--mean.pt" in names
        assert f"{COARSE_SET}--{POOLED_STRATUM}--mean.pt" in names
        assert f"{LEAD_SET}--{SPLIT}--mean.pt" in names

    def test_small_stratum_states_why_quality_is_unavailable(self, tmp_path: Path) -> None:
        """4-pair strata cannot fill 5 folds; the read says so instead of going quiet."""
        payload = run_axes(tmp_path, n_folds=5)
        matched = [
            read
            for read in cell_payload(payload)["quality"]
            if read["stimulus_set"] == COARSE_SET and read["stratum"] == MATCHED
        ]
        assert matched
        for read in matched:
            assert read["direction_accuracy"] is None
            assert "cannot fill 5" in read["unavailable_reason"]
            assert read["direction_norm"] > 0.0


class TestRefusals:
    def test_a_game_row_missing_cited_cells_is_refused(self, tmp_path: Path) -> None:
        """A game-set row with no citation regime is the older confounded rendering."""

        def strip_cited(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            rows[0] = {key: value for key, value in rows[0].items() if key != "cited_cells"}
            return rows

        stimuli_path, provenance_path, _ = write_corpus(tmp_path, mutate_provenance=strip_cited)
        with pytest.raises(ProvenanceError, match="older confounded rendering"):
            load_strata(provenance_path, load_stimuli(stimuli_path))

    def test_provenance_not_covering_the_corpus_is_refused(self, tmp_path: Path) -> None:
        def drop_one(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return rows[1:]

        stimuli_path, provenance_path, _ = write_corpus(tmp_path, mutate_provenance=drop_one)
        with pytest.raises(ProvenanceError, match="different renderings"):
            load_strata(provenance_path, load_stimuli(stimuli_path))

    def test_provenance_disagreeing_on_pairing_is_refused(self, tmp_path: Path) -> None:
        def swap_side(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            rows[0]["side"] = "B"
            return rows

        stimuli_path, provenance_path, _ = write_corpus(tmp_path, mutate_provenance=swap_side)
        with pytest.raises(ProvenanceError, match="disagrees with the stimulus corpus"):
            load_strata(provenance_path, load_stimuli(stimuli_path))

    def test_a_pair_straddling_strata_is_refused(self, tmp_path: Path) -> None:
        def straddle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            for row in rows:
                if row["id"] == f"{COARSE_SET}--p0--B":
                    row["cited_cells"] = SPLIT
            return rows

        stimuli_path, provenance_path, _ = write_corpus(tmp_path, mutate_provenance=straddle)
        with pytest.raises(ProvenanceError, match="straddles strata"):
            load_strata(provenance_path, load_stimuli(stimuli_path))

    def test_a_stratum_named_pooled_is_refused(self, tmp_path: Path) -> None:
        def rename(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            rows[0]["cited_cells"] = POOLED_STRATUM
            return rows

        stimuli_path, provenance_path, _ = write_corpus(tmp_path, mutate_provenance=rename)
        with pytest.raises(ProvenanceError, match="reserves"):
            load_strata(provenance_path, load_stimuli(stimuli_path))

    def test_an_edited_corpus_is_refused_end_to_end(self, tmp_path: Path) -> None:
        """The identity guard through this module's own entry point: edit the corpus, watch red."""
        stimuli_path, provenance_path, stimuli = write_corpus(tmp_path)
        root = tmp_path / "capture"
        write_planted_cell(root, stimuli, stimuli_digest(stimuli))
        stimuli_path.write_text(
            stimuli_path.read_text().replace("pair 0 side A", "pair 0 side A, edited")
        )
        args = build_parser().parse_args(
            [
                "--capture-root",
                str(root),
                "--stimuli",
                str(stimuli_path),
                "--provenance",
                str(provenance_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--positive-side",
                "A",
            ]
        )
        with pytest.raises(CellFormatError, match="different stimulus corpus"):
            run(args)
