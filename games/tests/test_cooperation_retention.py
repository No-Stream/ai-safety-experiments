"""CPU acceptance tests for the executable cooperation retention command."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from games import cooperation_retention as retention
from games.cooperation_interp import DisplacementRead, ProjectionRead
from games.evals import rebuild_summary
from games.interp_cells import CellIdentity, RowIndex, write_cell
from games.lora import adapter_config_identity
from reward_hacking.interp.jacobian import resolve_weights_identity


@pytest.fixture(autouse=True)
def _resolved_hub_identity(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retention, "resolve_weights_identity", lambda _model: "hf:test-base")


def _write_checkpoint(root: Path, step: int, *, base_model: str = "base-model") -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(f"{name}-{step}".encode())
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model,
                "peft_type": "LORA",
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["linear"],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(f"adapter-{step}".encode())
    return checkpoint


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_readouts(  # noqa: PLR0915 - one complete native authorization bundle
    root: Path,
    *,
    final_step: int = 3,
    base_model: str = "base-model",
    resolved_weights_identity: str = "hf:test-base",
) -> tuple[Path, ...]:
    checkpoint = root / f"checkpoint-{final_step}"
    if not checkpoint.exists():
        checkpoint = _write_checkpoint(root, final_step, base_model=base_model)
    adapter_weights_sha256 = _digest(checkpoint / "adapter_model.safetensors")
    adapter_config_sha256 = _digest(checkpoint / "adapter_config.json")
    adapter_config = json.loads(json.dumps(adapter_config_identity(checkpoint)))
    endpoint_specs = {
        "behavior": "game-behavior",
        "allocation": "self-report",
        "full-context": "self-report",
        "core-survey": "self-report",
        "prosocialness": "self-report",
        "local-dt": "dt-probes",
    }
    paths = [root / "final" / f"{endpoint}.summary.json" for endpoint in endpoint_specs]
    paths[0].parent.mkdir(parents=True)
    self_report_fields = {
        "item_id": "synthetic-item",
        "stem_digest": "synthetic",
        "family": "synthetic-family",
        "instrument": "synthetic-instrument",
        "subscale": "synthetic-subscale",
        "kind": "choice",
        "tier": "core",
        "wording": None,
        "twin_of": None,
        "counterpart": None,
        "counterpart_pair": None,
        "reverse_keyed": False,
        "scale_points": 2,
        "predicts_game": None,
        "option_labels": [],
        "presented_index": None,
        "canonical_index": None,
        "response": None,
        "score": None,
        "numeric": None,
        "tag": None,
        "announced_tag": None,
        "statement_matched_action": None,
        "chosen_label": None,
        "payoff_self": None,
        "payoff_other": None,
        "orientation": None,
        "option_order_name": "as-authored",
    }
    for path, section in zip(paths, endpoint_specs.values(), strict=True):
        endpoint = path.name.removesuffix(".summary.json")
        trace = path.with_name(f"{endpoint}.jsonl")
        meta = {
            "record": "meta",
            "sections": [section],
            "cooperation_endpoint": endpoint,
            "cooperation_request_count": 1,
            "cooperation_rendered_prompt_count": 1,
            "cooperation_rendered_prompt_digest": "a" * 64,
            "model_identity": retention.endpoint_model_identity(base_model),
            "model_weights_identity": retention.endpoint_model_identity(base_model),
            "model_adapter_dir": str(checkpoint.resolve()),
            "adapter_weights_sha256": adapter_weights_sha256,
            "adapter_config_sha256": adapter_config_sha256,
            "settings": {"arm": "care", "step": final_step},
        }
        body: dict[str, object] = {
            "record": section,
            "request_id": f"{endpoint}-request",
            "prompt_id": f"{endpoint}-prompt",
            "sample_index": 0,
            "parsed": True,
            "truncated_thinking": False,
        }
        if section == "game-behavior":
            body.update(game_id="twin-pd", cooperate=1.0)
        elif section == "self-report":
            body.update(self_report_fields)
        else:
            body.update(probe_id="synthetic-probe", source="local", theory=None)
        trace.write_text(f"{json.dumps(meta)}\n{json.dumps(body)}\n", encoding="utf-8")
        path.write_text(json.dumps(rebuild_summary(trace)), encoding="utf-8")

    capture = root / "capture" / "ladder-manifest.json"
    capture.parent.mkdir()
    capture.write_text(
        json.dumps(
            {
                "identity": {
                    "base_model": base_model,
                    "stimuli_sha256": "stimuli",
                    "rendered_sha256": "rendered",
                    "layer_convention": "post_block",
                    "n_layers": 2,
                    "hidden_size": 4,
                    "batch_size": 1,
                    "compute_dtype": "float32",
                    "store_dtype": "float32",
                    "stimulus_render": "verbatim",
                    "tokenizer_identity": "tokenizer",
                    "kernel_identity": "kernel",
                    "capture_prefix_states": True,
                    "prompt_end_rendered_sha256": "prompt-end",
                    "teacher_forced_rendered_sha256": "teacher-forced",
                    "natural_prefix_layers": [0],
                },
                "requested_cells": ["base/step-0", f"care/step-{final_step}"],
                "poolings": ["last"],
                "cells": {
                    "base/step-0": {"seconds": 1.0, "applied_adapter_weights": None},
                    f"care/step-{final_step}": {"seconds": 1.0, "applied_adapter_weights": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    identity_payload = CellIdentity(
        base_model=base_model,
        stimuli_sha256="stimuli",
        rendered_sha256="rendered",
        layer_convention="post_block",
        n_layers=2,
        hidden_size=4,
        batch_size=1,
        compute_dtype="float32",
        store_dtype="float32",
        stimulus_render="verbatim",
        tokenizer_identity="tokenizer",
        kernel_identity="kernel",
        capture_prefix_states=True,
        prompt_end_rendered_sha256="prompt-end",
        teacher_forced_rendered_sha256="teacher-forced",
        natural_prefix_layers=(0,),
    )
    rows = {
        "payoff": RowIndex(
            stimulus_ids=("p0-a", "p0-b"),
            sides=("A", "B"),
            pair_ids=("p0", "p0"),
            token_counts=(1, 1),
        )
    }
    base_activations = torch.zeros((2, 2, 4))
    write_cell(
        root / "capture" / "base" / "step-0",
        arm="base",
        step=0,
        identity=identity_payload,
        rows=rows,
        activations={("payoff", "last"): base_activations},
        applied_adapter_weights=None,
        adapter_weights_sha256=None,
        provenance={"adapter_dir": None, "adapter_config": None},
        prefix_activations={
            ("prompt_end", "last"): base_activations.clone(),
            ("teacher_forced", "last"): base_activations.clone(),
        },
        prefix_stimulus_ids=("p0-a", "p0-b"),
        natural_prefix_activations={
            "p0-a": torch.zeros((1, 1, 4)),
            "p0-b": torch.zeros((1, 1, 4)),
        },
    )
    write_cell(
        root / "capture" / "care" / f"step-{final_step}",
        arm="care",
        step=final_step,
        identity=identity_payload,
        rows=rows,
        activations={("payoff", "last"): base_activations + 1},
        applied_adapter_weights=2,
        adapter_weights_sha256=adapter_weights_sha256,
        provenance={"adapter_dir": str(checkpoint.resolve()), "adapter_config": adapter_config},
        prefix_activations={
            ("prompt_end", "last"): base_activations + 1,
            ("teacher_forced", "last"): base_activations + 2,
        },
        prefix_stimulus_ids=("p0-a", "p0-b"),
        natural_prefix_activations={
            "p0-a": torch.ones((1, 1, 4)),
            "p0-b": torch.ones((1, 1, 4)),
        },
    )

    geometry = root / "geometry" / "cooperation_interp.json"
    geometry.parent.mkdir()
    direction_paths: dict[str, dict[str, str]] = {"base": {}, "final": {}}
    for state, state_paths in direction_paths.items():
        for name in ("costly-other-regard", "decision-dependence", "trained-displacement"):
            direction = geometry.parent / "directions" / state / f"{name}.pt"
            direction.parent.mkdir(parents=True, exist_ok=True)
            torch.save({1: torch.ones(4)}, direction)
            state_paths[name] = str(direction.resolve())
    direction_manifest = geometry.parent / "direction-manifest.json"
    direction_manifest.write_text(json.dumps(direction_paths), encoding="utf-8")
    selected_target = geometry.parent / "selected-target.json"
    constructs = ("costly-other-regard", "decision-dependence")

    def projection(state: str, construct: str) -> dict[str, object]:
        return asdict(
            ProjectionRead(
                state=state,
                construct=construct,
                pooling="last",
                layer=1,
                fit_pair_ids=(f"{construct}-fit",),
                heldout_pair_ids=(f"{construct}-heldout",),
                fit_group_ids=(f"{construct}-fit-group",),
                heldout_group_ids=(f"{construct}-heldout-group",),
                fit_direction_norm=1.0,
                fit_direction_accuracy=0.5,
                fit_placebo_accuracy_mean=0.5,
                fit_placebo_accuracy_max=0.5,
                fit_accuracy_empirical_p=1.0,
                fit_split_half_cosine=0.0,
                heldout_positive_projection=0.0,
                heldout_negative_projection=0.0,
                heldout_projection_gap=0.0,
                shuffled_positive_projection=0.0,
                shuffled_negative_projection=0.0,
                shuffled_projection_gap=0.0,
                matched_norm_random_positive_projection=0.0,
                matched_norm_random_negative_projection=0.0,
                matched_norm_random_projection_gap=0.0,
                shuffled_direction_norm=1.0,
                matched_norm_random_direction_norm=1.0,
                grouped_projection_gaps={
                    "group=test": {"real": 0.0, "shuffled": 0.0, "matched_norm_random": 0.0}
                },
            )
        )

    def displacement(construct: str) -> dict[str, object]:
        return asdict(
            DisplacementRead(
                construct=construct,
                pooling="last",
                layer=1,
                pair_ids=(f"{construct}-heldout",),
                group_ids=(f"{construct}-heldout-group",),
                displacement_norm=0.0,
                displacement_projection=0.0,
                displacement_residual_norm=0.0,
                base_final_direction_cosine=0.0,
                displacement_base_direction_cosine=0.0,
                displacement_final_direction_cosine=0.0,
                base_axis_norm=1.0,
                final_axis_norm=1.0,
            )
        )

    natural_rows = [
        {
            "request_id": f"request-{request}",
            "base_stimulus_id": f"base-{request}",
            "final_stimulus_id": f"final-{request}",
            "layer": layer,
            "base_to_final_norm": 0.0,
            "base_to_final_alignment_with_trained_displacement": None,
        }
        for request in range(4)
        for layer in range(3)
    ]
    geometry.write_text(
        json.dumps(
            {
                "context": {
                    "capture_root": str(root / "capture"),
                    "cells": ["base/step-0", f"care/step-{final_step}"],
                    "cell_identities": {
                        "base/step-0": {
                            "identity": identity_payload.to_payload(),
                            "applied_adapter_weights": None,
                            "adapter_weights_sha256": None,
                        },
                        f"care/step-{final_step}": {
                            "identity": identity_payload.to_payload(),
                            "applied_adapter_weights": 2,
                            "adapter_weights_sha256": adapter_weights_sha256,
                        },
                    },
                    "direction_manifest": str(direction_manifest.resolve()),
                    "selected_target": str(selected_target.resolve()),
                    "export_pooling": "last",
                },
                "splits": {
                    construct: {
                        "fit_pair_ids": [f"{construct}-fit"],
                        "heldout_pair_ids": [f"{construct}-heldout"],
                        "fit_group_ids": [f"{construct}-fit-group"],
                        "heldout_group_ids": [f"{construct}-heldout-group"],
                        "group_by_pair": {
                            f"{construct}-fit": f"{construct}-fit-group",
                            f"{construct}-heldout": f"{construct}-heldout-group",
                        },
                    }
                    for construct in constructs
                },
                "confound_checks": {
                    "story_group": {"available": True, "rows": []},
                    "printed_position": {"available": True, "rows": []},
                    "procedure_regime": {"available": False, "reason": "synthetic fixture"},
                    "action_token_commitment": {"supported": True},
                },
                "projections": [
                    projection(state, construct)
                    for state in ("base", "final")
                    for construct in constructs
                ],
                "displacements": [displacement(construct) for construct in constructs],
                "calibration_selection": {},
                "natural_prefix_cross_check": {
                    "supported": True,
                    "content_confounding": {"present": False, "reasons": []},
                    "states": {
                        "base": {
                            "layers": [0, 1, 2],
                            "rows": [{"request_id": f"request-{i}"} for i in range(4)],
                        },
                        "final": {
                            "layers": [0, 1, 2],
                            "rows": [{"request_id": f"request-{i}"} for i in range(4)],
                        },
                    },
                    "base_to_final": {
                        "available": True,
                        "n_common_requests": 4,
                        "n_common_layers": 3,
                        "rows": natural_rows,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    selected_direction = Path(direction_paths["final"]["costly-other-regard"])
    selected_target.write_text(
        json.dumps(
            {
                "schema": "cooperation-generalization-selected-target/v1",
                "version": 1,
                "target_construct": "costly-other-regard",
                "direction": "costly-other-regard",
                "direction_path": str(selected_direction),
                "direction_sha256": _digest(selected_direction),
                "layer": 1,
                "magnitude": 1.0,
                "alpha_multiplier": 0.5,
                "calibration_metric": 0.0,
                "calibration_rationale": "synthetic fit-only calibration",
                "expected_effect": "synthetic expectation",
                "expectation_recorded_before_intervention": True,
                "geometry_report_path": str(geometry.resolve()),
                "geometry_report_sha256": _digest(geometry),
                "exploratory": True,
            }
        ),
        encoding="utf-8",
    )

    lens = root / "lens" / "cooperation_lens.json"
    lens.parent.mkdir()
    base_lens_path = root / "lens" / "base" / "lens.pt"
    final_lens_path = root / "lens" / "final" / "lens.pt"
    base_lens_path.parent.mkdir(parents=True)
    final_lens_path.parent.mkdir(parents=True)
    base_lens_path.write_bytes(b"base-lens")
    final_lens_path.write_bytes(b"final-lens")
    layer_quality = {
        "layer": 1,
        "relative_residual": 1.0,
        "explained_variance": 0.0,
        "n_samples": 1,
    }
    fit_quality = {
        "available": True,
        "n_eval_prompts_used": 1,
        "n_eval_prompts_skipped": 0,
        "positions_per_prompt_max": 1,
        "n_samples": 1,
        "jacobian": {"per_layer": [layer_quality]},
        "logit_lens_baseline": {"per_layer": [layer_quality]},
        "jacobian_beats_logit_lens": False,
        "median_residual_reduction_vs_logit_lens": 0.0,
    }
    layer_readout = {
        "quality": layer_quality,
        "direction_norm": 2.0,
        "random_norm": 2.0,
        "real": [{"token": "a", "logit": 0.0}],
        "matched_norm_random": [{"token": "b", "logit": 0.0}],
    }
    direction_readouts = {
        name: {"path": value, "sha256": _digest(Path(value)), "layers": {"1": layer_readout}}
        for name, value in direction_paths["base"].items()
    }
    lens_entry = {
        "readout_kind": "model-specific",
        "lens_state": "base",
        "applied_to_state": "base",
        "state": "base",
        "arm": "base",
        "step": 0,
        "model": base_model,
        "lens_path": str(base_lens_path.resolve()),
        "lens_acquisition": {"status": "complete"},
        "accumulation_identity": {"state": "base"},
        "fit_quality": fit_quality,
        "direction_readouts": direction_readouts,
    }
    final_entry = {
        **lens_entry,
        "readout_kind": "model-specific",
        "lens_state": "final",
        "applied_to_state": "final",
        "state": "final",
        "arm": "care",
        "step": final_step,
        "lens_path": str(final_lens_path.resolve()),
        "direction_readouts": {
            name: {
                "path": value,
                "sha256": _digest(Path(value)),
                "layers": {"1": layer_readout},
            }
            for name, value in direction_paths["final"].items()
        },
    }
    lens.write_text(
        json.dumps(
            {
                "base_model": base_model,
                "base_weights_identity": resolved_weights_identity,
                "tokenizer_content_sha256": "tokenizer",
                "corpus_identity_sha256": "corpus",
                "final_adapter": str(checkpoint.resolve()),
                "final_arm": "care",
                "final_step": final_step,
                "construct_capture": {
                    "cells": ["base/step-0", f"care/step-{final_step}"],
                    "final_adapter_weights_sha256": adapter_weights_sha256,
                },
                "states": {"base": lens_entry, "final": final_entry},
                "shared_base_lens_coordinate_sensitivity": {
                    "readout_kind": "shared-base-coordinate-sensitivity",
                    "approximation": True,
                    "lens_state": "base",
                    "applied_to_state": "final",
                    "cross_checkpoint_fidelity": fit_quality,
                    "direction_readouts": {
                        name: {
                            "path": value,
                            "sha256": _digest(Path(value)),
                            "layers": {"1": layer_readout},
                        }
                        for name, value in direction_paths["final"].items()
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    steering = root / "intervention" / "steering_records.jsonl"
    steering.parent.mkdir()
    adapter = {
        "path": str(checkpoint.resolve()),
        "config": adapter_config,
        "weights_filename": "adapter_model.safetensors",
        "weights_sha256": adapter_weights_sha256,
        "applied_adapter_weights": 2,
    }
    condition_specs = (
        ("none", "none"),
        ("costly-other-regard:L1:x0.5:steer:+", "steer:+"),
        ("costly-other-regard:L1:x0.5:placebo:+", "placebo:+"),
    )
    steering.write_text(
        "".join(
            json.dumps(
                {
                    "condition_key": condition_key,
                    "condition": condition,
                    "prompt_id": f"p{index}",
                    "response_text": "C",
                    "adapter": adapter,
                    "row_kind": "matrix",
                    "sample_index": 0,
                    "cooperate": True,
                    "truncated_thinking": False,
                    "label_print_order": "cooperate-first",
                    "deltanet_kernel": {"chunk_gated_delta_rule": "test.chunk"},
                }
            )
            + "\n"
            for index, (condition_key, condition) in enumerate(condition_specs)
        ),
        encoding="utf-8",
    )
    steering_records = [json.loads(line) for line in steering.read_text().splitlines()]
    grouped = {record["condition_key"]: 1 for record in steering_records}
    selected_target_payload = json.loads(selected_target.read_text())
    selected_target_wrapper = {
        "path": str(selected_target.resolve()),
        "sha256": _digest(selected_target),
        "payload": selected_target_payload,
        "geometry_report_path": str(geometry.resolve()),
        "geometry_report_sha256": _digest(geometry),
        "cell": {"direction": "costly-other-regard", "layer": 1, "alpha_multiplier": 0.5},
    }
    summary = {
        "command": "generate",
        "model_weights_identity": resolved_weights_identity,
        "seed": 0,
        "placebo_seed": 1,
        "n_rows": 1,
        "n_samples": 1,
        "max_new_tokens": 128,
        "selected_target": selected_target_wrapper,
        "adapter": adapter,
        "diagnostic_profile": "cooperation-generalization",
        "row_manifest": "private.json",
        "rendered_rows_sha256": "b" * 64,
        "resolved_sampler": {"temperature": 0.0},
        "conditions": retention.summarise_records(steering_records),
        "skipped_conditions": [],
    }
    (steering.parent / "steering_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (steering.parent / "steering_records.jsonl.resume.json").write_text(
        json.dumps({"identity": summary, "completed_units": grouped}), encoding="utf-8"
    )
    return (*paths, capture, geometry, lens, steering)


class TestReadoutValidation:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("adapter_weights_sha256", "0" * 64, "different final adapter weights"),
            ("adapter_config_sha256", "0" * 64, "different final adapter config"),
            ("model_adapter_dir", "other-checkpoint", "different final adapter path"),
            ("settings.step", 2, "disagree on final arm and step"),
        ],
    )
    def test_wrong_endpoint_checkpoint_identity_fails_before_pruning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        field: str,
        value: object,
        message: str,
    ) -> None:
        readouts = _write_readouts(tmp_path)
        trace = readouts[0].with_name("behavior.jsonl")
        lines = trace.read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        if field == "settings.step":
            meta["settings"]["step"] = value
        else:
            meta[field] = value
        trace.write_text(f"{json.dumps(meta)}\n{lines[1]}\n", encoding="utf-8")

        def pruning_must_not_run(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("invalid readout reached pruning")

        monkeypatch.setattr(retention, "prune_redundant_checkpoints", pruning_must_not_run)
        args = retention.build_parser().parse_args(
            [
                "--run-root",
                str(tmp_path),
                "--base-model",
                "base-model",
                "--retain-step",
                "3",
                *tuple(token for path in readouts for token in ("--readout", str(path))),
            ]
        )

        with pytest.raises(ValueError, match=message):
            retention.run_retention(args)

    def test_one_valid_readout_cannot_authorize_pruning(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, 3)
        readout = _write_readouts(tmp_path)[0]
        args = retention.build_parser().parse_args(
            [
                "--run-root",
                str(tmp_path),
                "--base-model",
                "base-model",
                "--retain-step",
                "3",
                "--readout",
                str(readout),
                "--dry-run",
            ]
        )

        with pytest.raises(ValueError, match="complete cooperation readout bundle"):
            retention.run_retention(args)

    def test_partial_endpoint_summary_cannot_authorize_pruning(self, tmp_path: Path) -> None:
        paths = _write_readouts(tmp_path)
        trace = paths[0].with_name("behavior.jsonl")
        lines = trace.read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        meta["cooperation_request_count"] = 2
        trace.write_text(f"{json.dumps(meta)}\n{lines[1]}\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"n_records.*request_count"):
            retention.validate_readout(paths[0])

    def test_missing_referenced_lens_cannot_authorize_pruning(self, tmp_path: Path) -> None:
        lens = _write_readouts(tmp_path)[8]
        payload = json.loads(lens.read_text(encoding="utf-8"))
        Path(payload["states"]["final"]["lens_path"]).unlink()

        with pytest.raises(FileNotFoundError, match="lens"):
            retention.validate_readout(lens)

    def test_missing_capture_tensor_cannot_authorize_pruning(self, tmp_path: Path) -> None:
        capture = _write_readouts(tmp_path)[6]
        (capture.parent / "care" / "step-3" / "activations.safetensors").unlink()

        with pytest.raises(FileNotFoundError, match="activations"):
            retention.validate_readout(capture)

    def test_complete_null_lens_readout_is_durable(self, tmp_path: Path) -> None:
        lens = _write_readouts(tmp_path)[8]
        payload = json.loads(lens.read_text(encoding="utf-8"))
        for state in payload["states"].values():
            state["fit_quality"]["jacobian_beats_logit_lens"] = False
            state["fit_quality"]["median_residual_reduction_vs_logit_lens"] = 0.0
        lens.write_text(json.dumps(payload), encoding="utf-8")

        retention.validate_readout(lens)

    def test_stale_selected_direction_cannot_authorize_pruning(self, tmp_path: Path) -> None:
        geometry = _write_readouts(tmp_path)[7]
        payload = json.loads(geometry.read_text(encoding="utf-8"))
        selection = json.loads(Path(payload["context"]["selected_target"]).read_text())
        torch.save({1: torch.zeros(4)}, Path(selection["direction_path"]))

        with pytest.raises(ValueError, match=r"direction_sha256.*does not match"):
            retention.validate_readout(geometry)

    def test_geometry_without_natural_measurement_cannot_authorize_pruning(
        self, tmp_path: Path
    ) -> None:
        geometry = _write_readouts(tmp_path)[7]
        payload = json.loads(geometry.read_text(encoding="utf-8"))
        del payload["natural_prefix_cross_check"]
        geometry.write_text(json.dumps(payload), encoding="utf-8")
        selected_target = Path(payload["context"]["selected_target"])
        selection = json.loads(selected_target.read_text(encoding="utf-8"))
        selection["geometry_report_sha256"] = _digest(geometry)
        selected_target.write_text(json.dumps(selection), encoding="utf-8")

        with pytest.raises(ValueError, match="natural_prefix_cross_check"):
            retention.validate_readout(geometry)

    def test_shape_only_lens_measurement_cannot_authorize_pruning(self, tmp_path: Path) -> None:
        lens = _write_readouts(tmp_path)[8]
        payload = json.loads(lens.read_text(encoding="utf-8"))
        payload["states"]["final"]["direction_readouts"]["costly-other-regard"]["layers"] = {}
        lens.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=r"layers.*nonempty object"):
            retention.validate_readout(lens)

    def test_steering_summary_is_rebuilt_from_records(self, tmp_path: Path) -> None:
        steering = _write_readouts(tmp_path)[9]
        summary_path = steering.with_name("steering_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        del summary["conditions"]["none"]["cooperate_rate"]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        with pytest.raises(ValueError, match="disagrees with its records"):
            retention.validate_readout(steering)

    def test_native_local_weight_identities_cover_nested_endpoint_tensors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        model = tmp_path / "model"
        (model / "nested").mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"top")
        (model / "nested" / "shard.safetensors").write_bytes(b"nested")
        resolved_identity = resolve_weights_identity(str(model))
        monkeypatch.setattr(retention, "resolve_weights_identity", resolve_weights_identity)
        readouts = _write_readouts(
            tmp_path,
            base_model=str(model),
            resolved_weights_identity=resolved_identity,
        )
        args = retention.build_parser().parse_args(
            [
                "--run-root",
                str(tmp_path),
                "--base-model",
                str(model),
                "--retain-step",
                "3",
                *tuple(token for path in readouts for token in ("--readout", str(path))),
                "--dry-run",
            ]
        )

        result = retention.run_retention(args)

        assert result.status == "dry-run"

    def test_partial_intervention_conditions_cannot_authorize_pruning(self, tmp_path: Path) -> None:
        steering = _write_readouts(tmp_path)[9]
        summary = steering.with_name("steering_summary.json")
        payload = json.loads(summary.read_text(encoding="utf-8"))
        payload["n_rows"] = 2
        summary.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=r"records.*are partial"):
            retention.validate_readout(steering)

    def test_native_readout_schemas_are_required(self, tmp_path: Path) -> None:
        paths = _write_readouts(tmp_path)

        for path in paths:
            retention.validate_readout(path)

        paths[8].write_text(json.dumps({"states": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="base_model"):
            retention.validate_readout(paths[8])

    def test_empty_or_arbitrary_readout_cannot_authorize_pruning(self, tmp_path: Path) -> None:
        empty = tmp_path / "final" / "report.md"
        empty.parent.mkdir()
        empty.write_text("# pending\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown native artifact"):
            retention.validate_readout(empty)

        arbitrary = tmp_path / "ready.json"
        arbitrary.write_text(json.dumps({"ready": True}), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown native artifact"):
            retention.validate_readout(arbitrary)

        renamed = tmp_path / "cooperation_lens.ready.json"
        renamed.write_text(json.dumps({"states": {"base": {}}}), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown native artifact"):
            retention.validate_readout(renamed)

    def test_native_readout_model_identity_is_bound_to_the_run(self, tmp_path: Path) -> None:
        readouts = _write_readouts(tmp_path, final_step=1)
        trace = readouts[0].with_name("behavior.jsonl")
        lines = trace.read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        meta["model_identity"] = "another-model"
        trace.write_text(f"{json.dumps(meta)}\n{lines[1]}\n", encoding="utf-8")
        args = retention.build_parser().parse_args(
            [
                "--run-root",
                str(tmp_path),
                "--base-model",
                "base-model",
                "--retain-step",
                "1",
                *tuple(token for path in readouts for token in ("--readout", str(path))),
                "--dry-run",
            ]
        )
        with pytest.raises(ValueError, match="different model identity"):
            retention.run_retention(args)


class TestRetentionCommand:
    def test_dry_run_checks_real_fixtures_without_loading_or_deleting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_checkpoint(tmp_path, 1)
        _write_checkpoint(tmp_path, 2)
        _write_checkpoint(tmp_path, 3)
        readouts = _write_readouts(tmp_path, final_step=3)
        output = tmp_path / "retention-plan.json"

        def model_load_must_not_run(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("dry-run loaded a model")

        monkeypatch.setattr(retention, "load_adapter_base", model_load_must_not_run)
        args = retention.build_parser().parse_args(
            [
                "--run-root",
                str(tmp_path),
                "--base-model",
                "base-model",
                "--retain-step",
                "2",
                "--retain-step",
                "3",
                "--adapter-step",
                "1",
                *tuple(token for path in readouts for token in ("--readout", str(path))),
                "--json-out",
                str(output),
                "--dry-run",
            ]
        )

        result = retention.run_retention(args)

        assert isinstance(result, retention.RetentionDryRun)
        assert result.adapter_steps == (1,)
        assert (tmp_path / "checkpoint-1").is_dir()
        assert (tmp_path / "checkpoint-2").is_dir()
        assert not (tmp_path / "retention_manifest.json").exists()
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "dry-run"

    def test_newest_incomplete_checkpoint_is_rejected_before_prune(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, 1)
        newest = _write_checkpoint(tmp_path, 2)
        (newest / "optimizer.pt").unlink()
        readouts = _write_readouts(tmp_path, final_step=2)
        args = retention.build_parser().parse_args(
            [
                "--run-root",
                str(tmp_path),
                "--base-model",
                "base-model",
                "--retain-step",
                "1",
                "--retain-step",
                "2",
                *tuple(token for path in readouts for token in ("--readout", str(path))),
                "--dry-run",
            ]
        )
        with pytest.raises(ValueError, match=r"optimizer[.]pt"):
            retention.run_retention(args)
        assert newest.is_dir()

    def test_production_path_passes_the_real_loader_and_validator_to_pruner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_checkpoint(tmp_path, 1)
        _write_checkpoint(tmp_path, 2)
        readouts = _write_readouts(tmp_path, final_step=2)
        observed: dict[str, object] = {}

        def fake_prune(  # noqa: PLR0913
            run_root: Path,
            *,
            retain_steps: tuple[int, ...],
            adapter_steps: tuple[int, ...],
            durable_readouts: tuple[Path, ...],
            load_checkpoint: object,
            validate_readout: object,
        ) -> retention.RetentionManifest:
            observed.update(
                run_root=run_root,
                retain_steps=retain_steps,
                adapter_steps=adapter_steps,
                durable_readouts=durable_readouts,
                load_checkpoint=load_checkpoint,
                validate_readout=validate_readout,
            )
            return retention.RetentionManifest(
                schema_version=1,
                status="complete",
                created_at="now",
                retained=(),
                adapter_retained=(),
                pruned=(),
                incomplete=(),
                checkpoint_bytes_before=0,
                checkpoint_bytes_after=0,
            )

        monkeypatch.setattr(retention, "prune_redundant_checkpoints", fake_prune)
        args = retention.build_parser().parse_args(
            [
                "--run-root",
                str(tmp_path),
                "--base-model",
                "base-model",
                "--retain-step",
                "1",
                "--retain-step",
                "2",
                *tuple(token for path in readouts for token in ("--readout", str(path))),
            ]
        )

        result = retention.run_retention(args)

        assert isinstance(result, retention.RetentionManifest)
        assert observed["retain_steps"] == (1, 2)
        authorization = tmp_path / "retention_readout_authorization.json"
        assert observed["durable_readouts"] == (authorization,)
        assert authorization.is_file()
        assert isinstance(observed["load_checkpoint"], retention.RuntimeAdapterCheckpointLoader)
        assert callable(observed["validate_readout"])

    def test_runtime_loader_reuses_real_base_and_repoints_adapter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        first = _write_checkpoint(tmp_path, 1)
        second = _write_checkpoint(tmp_path, 2)
        base = object()
        base_calls: list[str] = []
        attach_calls: list[tuple[Path, object | None]] = []

        def fake_load(base_model_id: str, **_kwargs: object) -> object:
            base_calls.append(base_model_id)
            return base

        def fake_attach(
            _base: object,
            checkpoint: Path,
            _base_model_id: str,
            *,
            existing: object | None = None,
        ) -> SimpleNamespace:
            attach_calls.append((checkpoint, existing))
            return SimpleNamespace(peft_model=object())

        monkeypatch.setattr(retention, "load_adapter_base", fake_load)
        monkeypatch.setattr(retention, "attach_adapter", fake_attach)
        loader = retention.RuntimeAdapterCheckpointLoader("base-model", device=torch.device("cpu"))

        loader(first)
        loader(second)

        assert base_calls == ["base-model"]
        assert [call[0] for call in attach_calls] == [first, second]
        assert attach_calls[0][1] is None
        assert attach_calls[1][1] is not None
