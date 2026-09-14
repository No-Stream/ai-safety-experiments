"""CPU-only integration checks for cooperation construct geometry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from games.cooperation_interp import (
    REPORT_FILENAME,
    SELECTION_FILENAME,
    CooperationInterpError,
    _construct_row_group,
    _natural_artifact_specs,
    _select_calibration_target,
    analyze_cached_cells,
    build_parser,
    load_intervention_expectation,
    load_natural_prefix_artifact,
    natural_prefix_cross_check,
    run,
)
from games.cooperation_lens import (
    DIRECTION_COSTLY_OTHER_REGARD,
    DIRECTION_DECISION_DEPENDENCE,
    REQUIRED_DIRECTIONS,
    load_direction_manifest,
)
from games.interp_axes import ProvenanceError, assert_no_pair_crosses_split, build_construct_splits
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    CapturedCell,
    CellFormatError,
    CellIdentity,
    Ladder,
    PairLayout,
    Stimulus,
    load_ladder,
    row_index_for,
    step_dir,
    stimuli_digest,
    write_cell,
    write_natural_prefix_capture,
)
from games.interp_displacement import displacement_vectors
from games.interp_steering import load_selected_target

CONSTRUCTS = (DIRECTION_COSTLY_OTHER_REGARD, DIRECTION_DECISION_DEPENDENCE)
FINAL_ARM = "cooperation-generalization-care-alpha-1"
FINAL_STEP = 20
N_LAYERS = 2
HIDDEN_SIZE = 8


def make_stimuli() -> list[Stimulus]:
    rows: list[Stimulus] = []
    for construct in CONSTRUCTS:
        for pair_number in range(4):
            split = "fit" if pair_number < 2 else "heldout"
            rows.extend(
                Stimulus(
                    stimulus_id=f"{construct}-{pair_number}-{side}",
                    stimulus_set=construct,
                    side=side,
                    pair_id=f"{construct}-{pair_number}",
                    text=f"opaque-{pair_number}-{side}",
                    assistant_prefix="opaque-prefix",
                    metadata={
                        "construct": construct,
                        "scenario_group": f"{construct}-group-{pair_number}",
                        "story_group": f"story-{pair_number % 2}",
                        "printed_position": "positive-first"
                        if pair_number % 2 == 0
                        else "positive-second",
                        "split": split,
                        "measurement_boundary": "pre_action",
                        "action_commitment_present": False,
                        **(
                            {
                                "dependence_mechanism": (
                                    "shared-randomness"
                                    if pair_number % 2 == 0 and side == "A"
                                    else "independent-randomness"
                                    if pair_number % 2 == 0
                                    else "shared-deterministic-procedure"
                                    if side == "A"
                                    else "independent-deterministic-procedure"
                                ),
                                "procedure_regime": (
                                    "stochastic" if pair_number % 2 == 0 else "deterministic"
                                ),
                            }
                            if construct == DIRECTION_DECISION_DEPENDENCE
                            else {}
                        ),
                    },
                )
                for side in ("A", "B")
            )
    return rows


def make_cell(stimuli: list[Stimulus], *, final: bool) -> CapturedCell:
    identity = CellIdentity(
        base_model="opaque-model",
        stimuli_sha256=stimuli_digest(stimuli),
        rendered_sha256="a" * 64,
        layer_convention="post_block",
        n_layers=N_LAYERS,
        hidden_size=HIDDEN_SIZE,
        batch_size=1,
        compute_dtype="float32",
        store_dtype="float32",
        stimulus_render="verbatim",
        tokenizer_identity="tokenizer-a",
        kernel_identity="kernel-a",
    )
    rows = {
        construct: row_index_for([row for row in stimuli if row.stimulus_set == construct], [4] * 8)
        for construct in CONSTRUCTS
    }
    activations: dict[tuple[str, str], torch.Tensor] = {}
    for construct_number, construct in enumerate(CONSTRUCTS):
        members = [row for row in stimuli if row.stimulus_set == construct]
        values = torch.zeros(len(members), N_LAYERS, HIDDEN_SIZE)
        for row_number, stimulus in enumerate(members):
            side_sign = 1.0 if stimulus.side == "A" else -1.0
            pair_number = int(stimulus.pair_id.rsplit("-", 1)[1])
            values[row_number, :, construct_number] = side_sign * (2.0 + pair_number / 10)
            values[row_number, :, 4] = pair_number / 10
            if final:
                values[row_number, 0, 5] += 1.0
                values[row_number, 1, 5] += 3.0
        activations[construct, "last"] = values
        activations[construct, "mean"] = -values
    return CapturedCell(
        arm=FINAL_ARM if final else BASE_ARM,
        step=FINAL_STEP if final else BASE_STEP,
        identity=identity,
        rows=rows,
        activations=activations,
        applied_adapter_weights=1 if final else None,
        adapter_weights_sha256="b" * 64 if final else None,
        provenance={"git_sha": "test"},
        source=Path(),
    )


def make_ladder(stimuli: list[Stimulus]) -> Ladder:
    return Ladder(cells=(make_cell(stimuli, final=False), make_cell(stimuli, final=True)))


def write_fixture(root: Path, stimuli: list[Stimulus]) -> None:
    for cell in make_ladder(stimuli).cells:
        write_cell(
            step_dir(root, cell.arm, cell.step),
            arm=cell.arm,
            step=cell.step,
            identity=cell.identity,
            rows=cell.rows,
            activations=cell.activations,
            applied_adapter_weights=cell.applied_adapter_weights,
            adapter_weights_sha256=cell.adapter_weights_sha256,
            provenance=cell.provenance,
            stimulus_metadata={row.stimulus_id: row.metadata for row in stimuli},
        )


def write_stimuli(path: Path, stimuli: list[Stimulus]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": row.stimulus_id,
                    "set": row.stimulus_set,
                    "side": row.side,
                    "pair_id": row.pair_id,
                    "text": row.text,
                    "assistant_prefix": row.assistant_prefix,
                    "metadata": row.metadata,
                }
            )
            for row in stimuli
        )
        + "\n"
    )


def test_cpu_cli_writes_hashed_geometry_and_selected_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stimuli = make_stimuli()
    capture_root = tmp_path / "capture"
    write_fixture(capture_root, stimuli)
    stimuli_path = tmp_path / "stimuli.jsonl"
    write_stimuli(stimuli_path, stimuli)
    reserved_path = tmp_path / "reserved.json"
    reserved_path.write_text(json.dumps({"group_ids": ["reserved"], "pair_ids": []}))
    monkeypatch.setattr("games.interp_cells.PRIVATE_ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr("games.cooperation_interp.PRIVATE_EXPECTATION_ROOTS", (tmp_path,))
    expectation_path = tmp_path / "expectation.json"
    expectation_path.write_text(
        json.dumps(
            {
                "target_construct": DIRECTION_COSTLY_OTHER_REGARD,
                "expected_effect": "opaque expected directional change",
                "expectation_recorded_before_intervention": True,
            }
        )
    )
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "--capture-root",
            str(capture_root),
            "--stimuli",
            str(stimuli_path),
            "--reserved-group-ids",
            str(reserved_path),
            "--intervention-expectation",
            str(expectation_path),
            "--out-dir",
            str(out_dir),
            "--pairs-per-construct",
            "4",
            "--layers",
            "0,1",
            "--poolings",
            "last,mean",
            "--n-placebos",
            "2",
        ]
    )
    payload = run(args)

    report_path = out_dir / REPORT_FILENAME
    selection = json.loads((out_dir / SELECTION_FILENAME).read_text())
    assert set(selection) == {
        "schema",
        "version",
        "target_construct",
        "direction",
        "direction_path",
        "direction_sha256",
        "layer",
        "magnitude",
        "alpha_multiplier",
        "calibration_metric",
        "calibration_rationale",
        "expected_effect",
        "expectation_recorded_before_intervention",
        "geometry_report_path",
        "geometry_report_sha256",
        "exploratory",
    }
    assert selection["schema"] == "cooperation-generalization-selected-target/v1"
    assert selection["version"] == 1
    assert selection["target_construct"] == DIRECTION_COSTLY_OTHER_REGARD
    assert selection["direction"] == DIRECTION_COSTLY_OTHER_REGARD
    assert selection["alpha_multiplier"] == 0.5
    assert selection["exploratory"] is True
    assert selection["expectation_recorded_before_intervention"] is True
    assert (
        selection["geometry_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    )
    assert selection["geometry_report_path"] == str(report_path.resolve())
    direction_path = Path(selection["direction_path"])
    assert selection["direction_sha256"] == hashlib.sha256(direction_path.read_bytes()).hexdigest()
    selected_directions = torch.load(direction_path, weights_only=True)
    selected_target = load_selected_target(
        out_dir / SELECTION_FILENAME,
        {selection["direction"]: selected_directions},
        direction_paths={selection["direction"]: direction_path},
    )
    assert selected_target["cell"]["alpha_multiplier"] == 0.5
    assert payload["confound_checks"]["story_group"]["supported"] is True
    assert payload["confound_checks"]["printed_position"]["supported"] is True
    assert payload["confound_checks"]["procedure_regime"]["supported"] is True
    assert payload["confound_checks"]["procedure_regime"]["pair_counts_by_construct_and_split"][
        DIRECTION_DECISION_DEPENDENCE
    ] == {
        "fit": {"deterministic": 1, "stochastic": 1},
        "heldout": {"deterministic": 1, "stochastic": 1},
    }
    assert payload["confound_checks"]["procedure_regime"]["heldout_projection_separation"]
    assert "residual confounding" in payload["confound_checks"]["procedure_regime"]["limitation"]
    assert payload["confound_checks"]["action_token_commitment"]["status"] == (
        "validated-pre-action"
    )
    assert (
        payload["context"]["cell_identities"][f"{FINAL_ARM}/step-{FINAL_STEP}"][
            "adapter_weights_sha256"
        ]
        == "b" * 64
    )
    assert (
        payload["context"]["cell_identities"]["base/step-0"]["identity"]["tokenizer_identity"]
        == "tokenizer-a"
    )
    assert payload["projections"][0]["grouped_projection_gaps"]
    direction_manifest = load_direction_manifest(out_dir / "direction-manifest.json")
    assert all(set(paths) == set(REQUIRED_DIRECTIONS) for paths in direction_manifest.values())
    exported = torch.load(
        out_dir / "directions" / "final" / f"{DIRECTION_COSTLY_OTHER_REGARD}.pt",
        weights_only=True,
    )
    assert exported[0][0].item() == pytest.approx(4.1)


def test_displacement_pooling_is_row_weighted() -> None:
    base = {"x": torch.zeros(6, 1, 1)}
    final = {"x": torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0, 5.0]).reshape(6, 1, 1)}
    layout = PairLayout(
        pair_ids=("p0", "p1", "p2"),
        positive_rows=torch.tensor([0, 2, 4]),
        negative_rows=torch.tensor([1, 3, 5]),
        positive_side="A",
        negative_side="B",
    )
    group = _construct_row_group("x", layout, torch.tensor([0, 1, 2]))
    assert displacement_vectors(final, base, group).full.item() == pytest.approx(7 / 3)
    stimuli = make_stimuli()
    result = analyze_cached_cells(
        make_ladder(stimuli), stimuli, pairs_per_construct=4, n_placebos=2
    )
    assert result.displacement_directions[0][5].item() == pytest.approx(1.0)
    assert result.displacement_directions[1][5].item() == pytest.approx(3.0)


def test_predeclared_construct_overrides_stronger_off_target_calibration(tmp_path: Path) -> None:
    stimuli = make_stimuli()
    result = analyze_cached_cells(
        make_ladder(stimuli), stimuli, pairs_per_construct=4, n_placebos=2
    )
    projections = tuple(
        replace(
            read,
            fit_direction_accuracy=(
                0.4 if read.construct == DIRECTION_COSTLY_OTHER_REGARD else 1.0
            ),
            fit_placebo_accuracy_max=(
                0.8 if read.construct == DIRECTION_COSTLY_OTHER_REGARD else 0.0
            ),
            fit_split_half_cosine=(0.1 if read.construct == DIRECTION_COSTLY_OTHER_REGARD else 1.0),
        )
        for read in result.projections
    )
    direction_paths: dict[str, dict[str, Path]] = {"final": {}}
    for construct in CONSTRUCTS:
        path = tmp_path / f"{construct}.pt"
        torch.save(result.directions["final"][construct], path)
        direction_paths["final"][construct] = path
    report_path = tmp_path / REPORT_FILENAME
    report_path.write_text("{}\n")

    selected = _select_calibration_target(
        projections,
        result.directions,
        direction_paths,
        preferred_pooling="last",
        geometry_report_path=report_path,
        geometry_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        expectation={
            "target_construct": DIRECTION_COSTLY_OTHER_REGARD,
            "expected_effect": "opaque expected directional change",
        },
    )

    assert selected["direction"] == DIRECTION_COSTLY_OTHER_REGARD
    assert selected["calibration_metric"] < 0.0


def test_intervention_expectation_rejects_unknown_target_construct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("games.cooperation_interp.PRIVATE_EXPECTATION_ROOTS", (tmp_path,))
    path = tmp_path / "expectation.json"
    path.write_text(
        json.dumps(
            {
                "target_construct": "unknown-construct",
                "expected_effect": "opaque expected directional change",
                "expectation_recorded_before_intervention": True,
            }
        )
    )
    with pytest.raises(ValueError, match="target_construct must be one of"):
        load_intervention_expectation(path)


def test_split_and_capture_identity_refusals() -> None:
    stimuli = make_stimuli()
    with pytest.raises(ValueError, match="reserved training/evaluation group"):
        build_construct_splits(
            stimuli,
            pairs_per_construct=4,
            reserved_groups=(stimuli[0].metadata["scenario_group"],),
        )
    crossed_groups = [
        replace(
            row,
            metadata={
                **row.metadata,
                "scenario_group": f"{row.stimulus_set}-group-0",
            },
        )
        if row.pair_id.endswith("-2")
        else row
        for row in stimuli
    ]
    with pytest.raises(ValueError, match="declares both"):
        build_construct_splits(crossed_groups, pairs_per_construct=4)
    malformed = [
        replace(row, metadata={**row.metadata, "action_commitment_present": True})
        if row is stimuli[0]
        else row
        for row in stimuli
    ]
    with pytest.raises(ValueError, match="action commitment"):
        analyze_cached_cells(make_ladder(malformed), malformed, pairs_per_construct=4)


def test_ladder_refuses_shuffled_rows_and_cropped_stimulus_digest(tmp_path: Path) -> None:
    stimuli = make_stimuli()
    root = tmp_path / "capture"
    write_fixture(root, stimuli)
    final_manifest = step_dir(root, FINAL_ARM, FINAL_STEP) / "manifest.json"
    manifest = json.loads(final_manifest.read_text())
    manifest["sets"][CONSTRUCTS[0]]["stimulus_ids"][:2] = reversed(
        manifest["sets"][CONSTRUCTS[0]]["stimulus_ids"][:2]
    )
    final_manifest.write_text(json.dumps(manifest))
    with pytest.raises(CellFormatError, match="different order"):
        load_ladder(root, stimuli_sha256=stimuli_digest(stimuli))

    clean_root = tmp_path / "clean"
    write_fixture(clean_root, stimuli)
    cropped = stimuli[:-1]
    with pytest.raises(CellFormatError, match="different stimulus corpus"):
        load_ladder(clean_root, stimuli_sha256=stimuli_digest(cropped))


def test_natural_prefix_pair_must_match_full_identity(tmp_path: Path) -> None:
    stimuli = make_stimuli()
    ladder = make_ladder(stimuli)
    paths: dict[str, Path] = {}
    for state, cell in zip(("base", "final"), ladder.cells, strict=True):
        state_rows = [
            Stimulus(
                f"natural-{state}",
                "natural",
                "A",
                "natural-0",
                f"opaque-{state}",
                metadata={
                    "natural_state": state,
                    "rollout_id": f"rollout-{state}",
                },
            )
        ]
        state_positions = [1] if state == "base" else [2, 3]
        selection_records = (
            {
                "stimulus_id": f"natural-{state}",
                "natural_state": state,
                "request_id": "request-0",
                "rollout_id": f"rollout-{state}",
                "scenario_group": "scenario-0",
                "positions": state_positions,
                "selection_rule": "opaque-rule",
            },
        )
        manifest_digest = "c" * 64
        selection_manifest: dict[str, Any] = {
            "version": 1,
            "rollout_ids_by_state": {
                "base": "rollout-base",
                "final": "rollout-final",
            },
            "request_ids": ["request-0"],
            "scenario_groups": ["scenario-0"],
            "selection_rule": "opaque-rule",
            "layers": [0],
            "path": f"private-{state}",
            "sha256": manifest_digest,
        }
        identity = replace(
            cell.identity,
            natural_prefix_layers=(0,),
            natural_selection_manifest_sha256=manifest_digest,
            natural_prefix_stimuli_sha256="",
            natural_prefix_rendered_sha256="",
        )
        paths[state] = write_natural_prefix_capture(
            tmp_path / state,
            identity=identity,
            stimuli=state_rows,
            activations={f"natural-{state}": torch.ones(len(state_positions), 1, HIDDEN_SIZE)},
            selection_records=selection_records,
            selection_manifest=selection_manifest,
            rendered_sha256=f"rendered-{state}",
            natural_state=state,
            arm=cell.arm,
            step=cell.step,
            applied_adapter_weights=cell.applied_adapter_weights,
            adapter_weights_sha256=cell.adapter_weights_sha256,
        )
    artifacts = {
        state: load_natural_prefix_artifact(paths[state], state=state, cell=cell)
        for state, cell in zip(("base", "final"), ladder.cells, strict=True)
    }
    with pytest.raises(CooperationInterpError, match="may not be swapped"):
        load_natural_prefix_artifact(paths["base"], state="final", cell=ladder.cells[1])
    result = analyze_cached_cells(ladder, stimuli, pairs_per_construct=4, n_placebos=2)
    cross_check = natural_prefix_cross_check(artifacts, result)
    assert cross_check["base_to_final"]["available"] is True
    assert cross_check["content_confounding"]["present"] is True
    assert cross_check["content_confounding"]["reasons"] == [
        "natural stimulus corpus digests differ",
        "rendered natural prefix digests differ",
        "selected token positions differ",
    ]

    payload = json.loads(paths["final"].read_text())
    payload["selection_manifest"]["request_ids"] = ["different"]
    paths["final"].write_text(json.dumps(payload))
    with pytest.raises(CooperationInterpError, match="do not match request_ids"):
        load_natural_prefix_artifact(paths["final"], state="final", cell=ladder.cells[1])

    payload["selection_manifest"]["request_ids"] = ["request-0", "request-0"]
    paths["final"].write_text(json.dumps(payload))
    with pytest.raises(CooperationInterpError, match="repeats or omits request_ids"):
        load_natural_prefix_artifact(paths["final"], state="final", cell=ladder.cells[1])
    with pytest.raises(ValueError, match="base and final"):
        _natural_artifact_specs([f"base={paths['base']}"])


def test_explicit_split_objects_reject_pair_overlap() -> None:
    stimuli = make_stimuli()
    splits = build_construct_splits(stimuli, pairs_per_construct=4)
    split = splits[CONSTRUCTS[0]]
    broken = split.__class__(
        construct=split.construct,
        group_by_pair=split.group_by_pair,
        fit_pair_ids=split.fit_pair_ids,
        heldout_pair_ids=(split.fit_pair_ids[0], *split.heldout_pair_ids),
    )
    with pytest.raises(ProvenanceError, match="both fit and held-out"):
        assert_no_pair_crosses_split(stimuli[:8], broken.fit_pair_ids, broken.heldout_pair_ids)
