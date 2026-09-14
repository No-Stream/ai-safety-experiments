"""Acceptance tests for the cooperation experiment's resumable phase wrapper.

The phase tests use CPU-only fake operations to exercise resume semantics without loading a model.
The plan tests inspect the real native commands and pass their arguments through the native parsers,
which catches a renamed flag before an expensive stage starts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from games import cooperation_sequence as plan
from games import train as game_train
from games.prompts import generate_prompt_rows
from games.stage_runner import Stage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _write(path: Path, contents: bytes = b"runtime input\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _complete_measurements() -> dict[str, object]:
    return {
        "throughput": {"generation": 1.0, "backward": 1.0, "optimizer": 1.0},
        "endpoints": {
            "final_eval": 1.0,
            "capture": 1.0,
            "lens": 1.0,
            "intervention": 1.0,
        },
        "headroom_fraction": 0.2,
        "prelaunch": {
            "smoke_0_6b": 1.0,
            "tiny_9b_update": 1.0,
            "tiny_9b_serving": 1.0,
            "tiny_9b_capture": 1.0,
            "tiny_9b_gradient": 1.0,
            "matching_sampler_screen": 1.0,
            "throughput_probe": 1.0,
            "baseline": 1.0,
        },
        "completion_token_cap": 32768,
        "planned_generation_completions": 2,
        "planned_generated_token_cap": 65536,
        "disk_actual_bytes": 1,
    }


@pytest.fixture(autouse=True)
def isolated_plan_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give every plan test a private runtime tree and a one-update research-sized plan."""
    for name in tuple(os.environ):
        if name.startswith("GAMES_COOP_"):
            monkeypatch.delenv(name, raising=False)
    runtime = tmp_path / "runtime"
    run = tmp_path / "run"
    monkeypatch.setenv("GAMES_COOP_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("GAMES_COOP_RUN_ROOT", str(run))
    monkeypatch.setenv("GAMES_COOP_MAX_STEPS", "1")
    monkeypatch.setenv("GAMES_COOP_COMPLETION_TOKENS", "32768")

    _write(
        runtime / "measurements.json",
        json.dumps(_complete_measurements()).encode(),
    )
    monkeypatch.setenv("GAMES_COOP_MEASUREMENTS", str(runtime / "measurements.json"))
    _write(
        run / "budget.json",
        json.dumps(_complete_measurements()).encode(),
    )
    monkeypatch.setenv("GAMES_COOP_BUDGET", str(run / "budget.json"))

    _write(runtime / "training.jsonl", b'{"id":"training-row"}\n')
    _write(
        runtime / "training-manifest.json",
        b'{"rows":[{"item_id":"training-row","family":"social-dilemma","split":"train"}]}\n',
    )
    _write(
        runtime / "evaluation.json",
        b'{"rows":[{"item_id":"evaluation-row","family":"social-dilemma","split":"eval"}]}\n',
    )
    _write(runtime / "interp-fit.jsonl", b'{"id":"fit-row"}\n')
    _write(runtime / "interp-quality.jsonl", b'{"id":"quality-row"}\n')
    _write(runtime / "interp-stimuli.jsonl", b'{"id":"stimulus-row"}\n')


@pytest.fixture
def minimal_prepare_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Install small, disjoint runtime inputs for the CPU prepare and print-plan paths."""
    runtime = tmp_path / "runtime"
    run = tmp_path / "run"
    survey = Path("games/data/survey").resolve()

    training_corpus = _write(
        runtime / "training.jsonl",
        b'{"id":"train-row-0"}\n{"id":"train-row-1"}\n{"id":"train-row-2"}\n',
    )
    training_manifest = _write(
        runtime / "training-manifest.json",
        json.dumps(
            {
                "row_count": 3,
                "corpus_path": training_corpus.name,
                "training_group_ids": ["train-group"],
            }
        ).encode(),
    )

    generated = generate_prompt_rows(
        "twin-pd", "care-alpha-1", split="eval", label_print_order="canonical"
    )[0]
    behavior_manifest = _write(
        runtime / "behavior-roster.json",
        json.dumps(
            {
                "elicitation": {
                    "forecast_template": "{target}",
                    "normative_choice_template": "{target}",
                    "normative_numeric_template": "{target}",
                    "matrix_target_template": "{coop_label}",
                    "numeric_targets": {},
                    "normative_target": "target",
                    "normative_dimension": "welfare",
                    "construct": "x",
                    "expected_direction": "x",
                },
                "pairs": [
                    {
                        "pair_id": "behavior-pair",
                        "family": "social-dilemma",
                        "game_id": "twin-pd",
                        "grading": "care-alpha-1",
                        "scenario_id": generated["reskin_id"],
                        "payoff_variant": generated["payoff_variant"],
                        "scenario_group": "behavior-group",
                        "label_print_orders": ["canonical", "swapped"],
                        "samples": 2,
                    }
                ],
                "allocation": {
                    "diagnostic_id": "fixture-allocation",
                    "stem": "Choose one option.",
                    "option_payoffs": [[10, 0], [9, 8], [7, 7]],
                    "samples": 2,
                },
            }
        ).encode(),
    )

    def write_stimuli(path: Path, prefix: str, groups: tuple[str, ...]) -> Path:
        rows = [
            {
                "id": f"{prefix}-{index}",
                "pair_id": f"{prefix}-pair-{index}",
                "metadata": {"scenario_group": group},
            }
            for index, group in enumerate(groups)
        ]
        return _write(path, ("\n".join(json.dumps(row) for row in rows) + "\n").encode())

    construct_stimuli = write_stimuli(
        runtime / "construct-stimuli.jsonl", "construct", ("construct-group-0", "construct-group-1")
    )
    lens_fit_measurement = write_stimuli(
        runtime / "lens-fit-measurement.jsonl", "lens-fit", ("lens-group-fit",)
    )
    lens_quality = write_stimuli(
        runtime / "lens-quality.jsonl", "lens-quality", ("lens-group-quality",)
    )
    lens_fit_smoke = write_stimuli(
        runtime / "lens-fit-smoke.jsonl", "lens-smoke", ("lens-smoke-group",)
    )
    steering_rows = _write(
        runtime / "steering-diagnostic-rows.json", b'{"rows":[{"id":"steer-0"}]}\n'
    )
    intervention_expectation = _write(
        runtime / "intervention-expectation.json",
        b'{"expected_effect":"directional probe","expectation_recorded_before_intervention":true}\n',
    )
    natural_selection = _write(
        runtime / "natural-prefix-selection.json",
        b'{"layers":[8,16,24],"request_ids":["one","two","three","four"],'
        b'"rollout_ids_by_state":{"base":"base/step-0",'
        b'"final":"cooperation-generalization-care-alpha-1/step-2"}}\n',
    )
    model_metadata = _write(
        runtime / "model-metadata.json",
        json.dumps(
            {
                "parameter_count": 100,
                "hidden_size": 4,
                "n_layers": 3,
                "source_layers": 2,
                "dtype": "bfloat16",
                "lora_rank": 2,
                "lora_targets": [
                    {"name": "first", "count": 2, "in_features": 3, "out_features": 4},
                    {"name": "second", "count": 1, "in_features": 2, "out_features": 5},
                ],
            }
        ).encode(),
    )
    measurements = _write(
        run / "measurements.json",
        json.dumps(_complete_measurements()).encode(),
    )
    reserved = _write(
        runtime / "reserved-group-ids.json",
        json.dumps(
            {
                "group_ids": [
                    "train-group",
                    "behavior-group",
                    "construct-group-0",
                    "construct-group-1",
                    "lens-group-fit",
                    "lens-group-quality",
                ],
                "pair_ids": [
                    "behavior-pair",
                    "construct-pair-0",
                    "construct-pair-1",
                    "lens-fit-pair-0",
                    "lens-quality-pair-0",
                ],
            }
        ).encode(),
    )

    env_paths = {
        "RUNTIME_ROOT": runtime,
        "RUN_ROOT": run,
        "TRAIN_CORPUS": training_corpus,
        "TRAIN_MANIFEST": training_manifest,
        "BEHAVIOR_MANIFEST": behavior_manifest,
        "CONSTRUCT_STIMULI": construct_stimuli,
        "LENS_FIT_SMOKE": lens_fit_smoke,
        "LENS_FIT_MEASUREMENT": lens_fit_measurement,
        "LENS_QUALITY": lens_quality,
        "STEERING_ROWS": steering_rows,
        "INTERVENTION_EXPECTATION": intervention_expectation,
        "NATURAL_SELECTION": natural_selection,
        "MODEL_METADATA": model_metadata,
        "MEASUREMENTS": measurements,
        "BUDGET": measurements,
        "RESERVED_GROUP_IDS": reserved,
    }
    for suffix, path in env_paths.items():
        monkeypatch.setenv(f"GAMES_COOP_{suffix}", str(path))
    monkeypatch.setenv("GAMES_COOP_SURVEY_DATA_DIR", str(survey))
    monkeypatch.setenv("GAMES_COOP_MAX_STEPS", "2")
    monkeypatch.setenv("GAMES_COOP_PROMPTS_PER_STEP", "2")
    monkeypatch.setenv("GAMES_COOP_GROUP", "2")
    monkeypatch.setenv("GAMES_COOP_OVERSAMPLE", "1")
    model_snapshot = tmp_path / "qwen3.5-9b-snapshot"
    smoke_snapshot = tmp_path / "qwen3-0.6b-snapshot"
    model_snapshot.mkdir()
    smoke_snapshot.mkdir()
    monkeypatch.setenv("GAMES_COOP_MODEL_PATH", str(model_snapshot))
    monkeypatch.setenv("GAMES_COOP_SMOKE_MODEL_PATH", str(smoke_snapshot))
    monkeypatch.setenv("GAMES_COOP_JLENS_ROOT", str(tmp_path / "jlens"))
    (tmp_path / "jlens").mkdir()
    from games import (  # noqa: PLC0415
        cooperation_budget,
        cooperation_evals,
    )

    monkeypatch.setattr(
        cooperation_budget,
        "count_behavior_inputs",
        lambda _path: cooperation_budget.BehaviorInputCounts(
            pairs=1,
            rendered_prompts=2,
            completions=4,
            by_family={"social-dilemma": 4},
            context_responses=4,
            allocation_responses=1,
        ),
    )
    monkeypatch.setattr(cooperation_evals, "validate_experiment_coverage", lambda _roster: None)
    monkeypatch.setattr(
        cooperation_evals,
        "build_allocation_plan",
        lambda _roster, **_kwargs: (),
    )
    monkeypatch.setattr(
        cooperation_budget,
        "count_survey_inputs",
        lambda _path: cooperation_budget.SurveyInputCounts(
            core_responses=1,
            prosocialness_responses=1,
            decision_theory_responses=1,
        ),
    )
    return env_paths


def _phase_specs() -> tuple[plan.Phase, ...]:
    phases = tuple(plan.phase_specs())
    assert phases, "the cooperation plan must contain at least one phase"
    return phases


class TestSnapshotReadiness:
    def test_complete_indexed_snapshot_requires_all_unique_referenced_shards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = tmp_path / "indexed-snapshot"
        snapshot.mkdir()
        _write(
            snapshot / "model.safetensors.index.json",
            json.dumps(
                {
                    "weight_map": {
                        "layer.0": "model.safetensors-00001-of-00002.safetensors",
                        "layer.1": "model.safetensors-00001-of-00002.safetensors",
                        "layer.2": "model.safetensors-00002-of-00002.safetensors",
                    }
                }
            ).encode(),
        )
        _write(snapshot / "model.safetensors-00001-of-00002.safetensors", b"shard one")
        _write(snapshot / "model.safetensors-00002-of-00002.safetensors", b"shard two")
        monkeypatch.setenv("GAMES_COOP_MODEL_PATH", str(snapshot))

        assert plan._weights_present()

    def test_missing_indexed_shard_rejects_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = tmp_path / "indexed-snapshot"
        snapshot.mkdir()
        _write(
            snapshot / "model.safetensors.index.json",
            json.dumps(
                {
                    "weight_map": {
                        "layer.0": "model.safetensors-00001-of-00002.safetensors",
                        "layer.1": "model.safetensors-00002-of-00002.safetensors",
                    }
                }
            ).encode(),
        )
        _write(snapshot / "model.safetensors-00001-of-00002.safetensors", b"shard one")
        monkeypatch.setenv("GAMES_COOP_MODEL_PATH", str(snapshot))

        assert not plan._weights_present()

    def test_oversized_unrelated_safetensor_cannot_pass_without_exact_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = tmp_path / "unrelated-snapshot"
        snapshot.mkdir()
        unrelated = snapshot / "unrelated.safetensors"
        with unrelated.open("wb") as handle:
            handle.truncate(20_000_000_001)
        monkeypatch.setenv("GAMES_COOP_MODEL_PATH", str(snapshot))

        assert not plan._weights_present()

    def test_unindexed_snapshot_accepts_one_exact_model_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = tmp_path / "single-file-snapshot"
        snapshot.mkdir()
        _write(snapshot / "model.safetensors", b"single complete file")
        monkeypatch.setenv("GAMES_COOP_MODEL_PATH", str(snapshot))

        assert plan._weights_present()


def _phase(phases: Sequence[plan.Phase], name: str) -> plan.Phase:
    matches = [phase for phase in phases if name in phase.name.lower()]
    assert len(matches) == 1, f"expected one phase containing {name!r}, got {matches}"
    return matches[0]


def _phase_one_of(phases: Sequence[plan.Phase], *names: str) -> plan.Phase:
    for name in names:
        matches = [phase for phase in phases if name in phase.name.lower()]
        if len(matches) == 1:
            return matches[0]
    raise AssertionError(f"expected one phase containing one of {names!r}, got {phases}")


def _operations(phases: Sequence[plan.Phase]) -> list[plan.Operation]:
    return [operation for phase in phases for operation in phase.operations]


def _module_args(argv: Sequence[str]) -> tuple[str, list[str]] | None:
    """Extract a Python module command from a native operation, if it has one."""
    if "-m" not in argv:
        return None
    module_index = argv.index("-m")
    assert module_index + 1 < len(argv)
    return argv[module_index + 1], list(argv[module_index + 2 :])


def _parse_cooperation_corpus(args: list[str]) -> Any:
    from games.cooperation_corpus import _parse_args  # noqa: PLC0415

    return _parse_args(args)


def _parse_cooperation_screen(args: list[str]) -> Any:
    from games.cooperation_screen import _parse_args  # noqa: PLC0415

    return _parse_args(args)


def _parse_cooperation_eval_runner(args: list[str]) -> Any:
    from games.cooperation_eval_runner import build_parser  # noqa: PLC0415

    return build_parser().parse_args(args)


def _parse_cooperation_budget(args: list[str]) -> Any:
    from games.cooperation_budget import build_parser  # noqa: PLC0415

    return build_parser().parse_args(args)


def _parse_cooperation_retention(args: list[str]) -> Any:
    from games.cooperation_retention import build_parser  # noqa: PLC0415

    return build_parser().parse_args(args)


def _parse_cooperation_interp(args: list[str]) -> Any:
    from games.cooperation_interp import build_parser  # noqa: PLC0415

    return build_parser().parse_args(args)


def _parse_train(args: list[str]) -> Any:
    from games.train import _parse_args  # noqa: PLC0415

    return _parse_args(args)


def _parse_throughput_probe(args: list[str]) -> Any:
    from games.throughput_probe import parse_args  # noqa: PLC0415

    return parse_args(args)


def _parse_run_evals(args: list[str]) -> Any:
    from games.run_evals import _parse_args  # noqa: PLC0415

    return _parse_args(args)


def _parse_interp_capture(args: list[str]) -> Any:
    from games.interp_capture import build_parser  # noqa: PLC0415

    return build_parser().parse_args(args)


def _parse_cooperation_lens(args: list[str]) -> Any:
    from games.cooperation_lens import build_parser  # noqa: PLC0415

    return build_parser().parse_args(args)


def _parse_interp_steering(args: list[str]) -> Any:
    from games.interp_steering import build_parser  # noqa: PLC0415

    return build_parser().parse_args(args)


def _parse_lens_fit_gate(args: list[str]) -> Any:
    from reward_hacking.interp.lens_fit_gate import _parse_args  # noqa: PLC0415

    return _parse_args(args)


def _parse_cooperation_sequence(args: list[str]) -> Any:
    from games.cooperation_sequence import build_parser  # noqa: PLC0415

    return build_parser().parse_args(args)


NATIVE_PARSERS: dict[str, Callable[[list[str]], Any]] = {
    "games.cooperation_corpus": _parse_cooperation_corpus,
    "games.cooperation_screen": _parse_cooperation_screen,
    "games.cooperation_eval_runner": _parse_cooperation_eval_runner,
    "games.cooperation_budget": _parse_cooperation_budget,
    "games.cooperation_retention": _parse_cooperation_retention,
    "games.cooperation_interp": _parse_cooperation_interp,
    "games.train": _parse_train,
    "games.throughput_probe": _parse_throughput_probe,
    "games.run_evals": _parse_run_evals,
    "games.interp_capture": _parse_interp_capture,
    "games.cooperation_lens": _parse_cooperation_lens,
    "games.interp_steering": _parse_interp_steering,
    "reward_hacking.interp.lens_fit_gate": _parse_lens_fit_gate,
    "games.cooperation_sequence": _parse_cooperation_sequence,
}


def _parse_with_native_parser(module: str, args: list[str]) -> Any:
    """Parse generated arguments through the target module's own parser."""
    parser = NATIVE_PARSERS.get(module)
    if parser is None:
        raise AssertionError(f"no native parser adapter for generated module {module!r}")
    return parser(args)


class TestGeneratedCommands:
    def test_every_native_operation_is_accepted_by_its_real_parser(self) -> None:
        """A plan flag is executable data, so the parser is the source of truth for its flags."""
        for operation in _operations(_phase_specs()):
            command = _module_args(operation.argv)
            assert command is not None, f"operation {operation.name!r} has no Python module command"
            module, args = command
            _parse_with_native_parser(module, args)

    def test_smoke_uses_the_0_6b_plumbing_checkpoint(self) -> None:
        smoke = _phase(_phase_specs(), "0.6b")
        argv = [token for operation in smoke.operations for token in operation.argv]

        assert plan.smoke_model_source() in argv
        assert Path(plan.smoke_model_source()).is_absolute()

    def test_smoke_command_matches_the_effective_native_training_config(self) -> None:
        from games.train import _config_from_namespace  # noqa: PLC0415

        smoke = _phase(_phase_specs(), "0.6b")
        train_operation = next(
            operation
            for operation in smoke.operations
            if _module_args(operation.argv)[0] == "games.train"  # type: ignore[index]
        )
        _, args = _module_args(train_operation.argv)  # type: ignore[misc]
        config = _config_from_namespace(_parse_train(args))

        assert config.max_steps == 3
        assert config.num_generations == 4
        assert config.prompts_per_step == 2
        assert config.max_completion_tokens == 2048
        all_arguments = [token for operation in smoke.operations for token in operation.argv]
        assert f"checkpoint-{config.max_steps}" in " ".join(all_arguments)

    def test_0_6b_smoke_does_not_claim_qwen3_5_interp_provenance(self) -> None:
        smoke = _phase(_phase_specs(), "0.6b")
        modules = {
            command[0]
            for operation in smoke.operations
            if (command := _module_args(operation.argv)) is not None
        }

        assert modules.isdisjoint(
            {
                "games.interp_capture",
                "games.cooperation_interp",
                "reward_hacking.interp.lens_fit_gate",
                "games.cooperation_lens",
                "games.interp_steering",
            }
        )

    def test_research_update_capture_and_gradient_are_tiny_9b_phases(self) -> None:
        phases = _phase_specs()
        for names in (("update", "train"), ("capture",), ("gradient",)):
            phase = _phase_one_of(phases, *names)
            argv = [token for operation in phase.operations for token in operation.argv]
            assert plan.model_source() in argv, phase.name
            assert phase.needs_gpu is True

    def test_research_weight_loaders_share_one_immutable_local_snapshot(self) -> None:
        research_modules = {
            "games.interp_capture",
            "games.cooperation_interp",
            "reward_hacking.interp.lens_fit_gate",
            "games.cooperation_lens",
            "games.interp_steering",
        }
        sources: set[str] = set()
        seen_modules: set[str] = set()
        for phase in _phase_specs():
            if "smoke" in phase.name.lower():
                continue
            for operation in phase.operations:
                command = _module_args(operation.argv)
                assert command is not None
                module, args = command
                if module not in research_modules:
                    continue
                seen_modules.add(module)
                for flag in ("--model", "--model-id", "--base-model"):
                    if flag in args:
                        sources.add(args[args.index(flag) + 1])
        assert seen_modules == research_modules
        assert sources == {plan.model_source()}
        assert Path(plan.model_source()).is_absolute()

    def test_print_metadata_keeps_logical_model_id_separate_from_snapshot(
        self, minimal_prepare_inputs: dict[str, Path]
    ) -> None:
        del minimal_prepare_inputs

        metadata = plan.static_plan()

        assert metadata["model"] == plan.model_id()
        assert metadata["model"] == "Qwen/Qwen3.5-9B"
        assert metadata["model_source"] == plan.model_source()

    def test_research_commands_name_both_the_logical_model_and_load_source(self) -> None:
        sampler_modules = {
            "games.train",
            "games.throughput_probe",
            "games.cooperation_screen",
            "games.cooperation_eval_runner",
        }
        seen: set[str] = set()
        for operation in _operations(_phase_specs()):
            command = _module_args(operation.argv)
            assert command is not None
            module, args = command
            if module not in sampler_modules or plan.model_source() not in args:
                continue
            seen.add(module)
            separates_source = module in {"games.train", "games.throughput_probe"}
            source_flag = "--model-source" if separates_source else "--model"
            identity_flag = "--model" if separates_source else "--model-id"
            assert args[args.index(source_flag) + 1] == plan.model_source()
            assert args[args.index(identity_flag) + 1] == plan.model_id()
            if module == "games.train":
                config = game_train._parse_args(args)
                assert config.model_id == plan.model_id()
                assert config.load_source == plan.model_source()
                assert config.max_completion_tokens == 32768
            if module == "games.throughput_probe":
                from games.throughput_probe import parse_args  # noqa: PLC0415

                probe = parse_args(args)
                assert probe.config.model_id == plan.model_id()
                assert probe.config.load_source == plan.model_source()
                assert probe.config.max_completion_tokens == 32768

        assert seen == sampler_modules

    def test_behavior_evaluations_use_vllm(self) -> None:
        phases = _phase_specs()
        for name in ("baseline", "final"):
            phase = _phase(phases, name)
            argv = [token for operation in phase.operations for token in operation.argv]
            assert argv[argv.index("--backend") + 1] == "vllm"

    def test_research_generation_caps_are_32768(self) -> None:
        phases = _phase_specs()
        for names in (
            ("throughput",),
            ("baseline",),
            ("update", "train"),
            ("final",),
            ("intervention",),
        ):
            phase = _phase_one_of(phases, *names)
            argv = [token for operation in phase.operations for token in operation.argv]
            cap_flags = [
                flag for flag in ("--max-completion-tokens", "--max-new-tokens") if flag in argv
            ]
            assert cap_flags, f"{phase.name} must carry its explicit completion cap"
            for flag in cap_flags:
                assert argv[argv.index(flag) + 1] == "32768", phase.name

    def test_training_persists_every_checkpoint_and_retention_manifest(self) -> None:
        train = _phase(_phase_specs(), "research training")
        argv = [token for operation in train.operations for token in operation.argv]

        assert argv[argv.index("--save-steps") + 1] == "1"
        assert argv[argv.index("--save-total-limit") + 1] == "0"
        assert "--record-retention-manifest" in argv

    def test_budget_precedes_training(self) -> None:
        phases = _phase_specs()
        phase_names = [phase.name.lower() for phase in phases]
        budget_index = next(index for index, name in enumerate(phase_names) if "budget" in name)
        train_index = next(
            index for index, name in enumerate(phase_names) if "research training" in name
        )

        assert budget_index < train_index

    def test_budget_requires_every_endpoint_cost(self, tmp_path: Path) -> None:
        complete = _complete_measurements()
        measured = _write(tmp_path / "measurements.json", json.dumps(complete).encode())

        assert plan.measured_budget_is_complete(measured)
        measurement_endpoints = tuple(cast("dict[str, float]", complete["endpoints"]))
        for missing in measurement_endpoints:
            incomplete = _complete_measurements()
            endpoints = cast("dict[str, float]", incomplete["endpoints"])
            del endpoints[missing]
            path = _write(
                tmp_path / f"missing-{missing}.json",
                json.dumps(incomplete).encode(),
            )
            assert not plan.measured_budget_is_complete(path), missing

        for endpoint in measurement_endpoints:
            for value in (0.0, -1.0, math.nan, True):
                invalid = _complete_measurements()
                endpoints = cast("dict[str, float]", invalid["endpoints"])
                endpoints[endpoint] = value
                path = _write(
                    tmp_path / f"invalid-{endpoint}-{str(value).replace('.', '_')}.json",
                    json.dumps(invalid, allow_nan=True).encode(),
                )
                assert not plan.measured_budget_is_complete(path), (endpoint, value)

    def test_handwritten_endpoint_marker_cannot_authorize_training(self, tmp_path: Path) -> None:
        marker = _write(
            tmp_path / "handwritten-marker.json",
            json.dumps(
                {
                    "all_endpoints_measured": True,
                    "endpoints": dict.fromkeys(
                        cast("dict[str, float]", _complete_measurements()["endpoints"]), 1.0
                    ),
                }
            ).encode(),
        )

        assert not plan.measured_budget_is_complete(marker)


class TestCpuPreparation:
    def test_membership_tracks_disjoint_train_behavior_construct_and_lens_groups(
        self, minimal_prepare_inputs: dict[str, Path]
    ) -> None:
        del minimal_prepare_inputs

        memberships = plan.build_membership()

        assert set(memberships) == {"train", "behavior", "construct", "lens"}
        assert memberships["train"]["group_ids"] == ["train-group"]
        assert memberships["behavior"]["group_ids"] == [
            "allocation::fixture-allocation",
            "behavior-group",
        ]
        assert memberships["behavior"]["pair_ids"] == [
            "behavior-pair",
            "fixture-allocation",
        ]
        assert memberships["construct"]["group_ids"] == [
            "construct-group-0",
            "construct-group-1",
        ]
        assert memberships["lens"]["group_ids"] == ["lens-group-fit", "lens-group-quality"]

        all_groups = {
            group for membership in memberships.values() for group in membership["group_ids"]
        }
        assert all_groups == {
            "train-group",
            "behavior-group",
            "allocation::fixture-allocation",
            "construct-group-0",
            "construct-group-1",
            "lens-group-fit",
            "lens-group-quality",
        }

    def test_static_plan_uses_runtime_counts_and_model_metadata_for_disk_bytes(
        self, minimal_prepare_inputs: dict[str, Path]
    ) -> None:
        del minimal_prepare_inputs

        payload = plan.static_plan()
        training = payload["training"]
        generation = payload["generation"]
        disk = payload["disk"]
        assert isinstance(training, dict)
        assert isinstance(generation, dict)
        assert isinstance(disk, dict)

        assert training["manifest_rows"] == 3
        assert training["corpus_rows"] == 3
        assert generation["training_completions"] == 8
        assert generation["matching_screen_completions"] == 24
        assert disk["base_model_bf16_bytes"] == 200
        assert disk["lora_parameter_count"] == 42
        assert disk["checkpoint_bytes_total"] == 1260
        assert disk["estimated_bytes"] == 3380

    def test_print_plan_emits_runtime_counts_and_numerical_disk_estimate(
        self,
        minimal_prepare_inputs: dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        del minimal_prepare_inputs

        expected = plan.static_plan()
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out

        assert "unknown (manifest missing)" not in printed
        assert str(expected["training"]["corpus_rows"]) in printed  # type: ignore[index]
        assert str(expected["generation"]["training_completions"]) in printed  # type: ignore[index]
        assert str(expected["disk"]["estimated_bytes"]) in printed  # type: ignore[index]
        assert "natural-prefix requests: 4" in printed
        assert "natural-prefix layers: 8,16,24" in printed
        assert "scripts/tmux_run.sh cooperation-generalization-smoke-0_6b" in printed
        assert "scripts/resource-limits.sh --gpu -t 45m" in printed

    def test_long_plan_requires_a_persisted_measured_step_cap(
        self, minimal_prepare_inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        measurements_path = minimal_prepare_inputs["MEASUREMENTS"]
        monkeypatch.setenv("GAMES_COOP_MAX_STEPS", "21")

        with pytest.raises(ValueError, match="measured step cap"):
            plan.max_steps()

        measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        measurements["measured_max_steps"] = 21
        measurements["completion_token_cap"] = 32768
        measurements["planned_generation_completions"] = 1
        measurements["planned_generated_token_cap"] = 32768
        measurements["disk_actual_bytes"] = 1
        _write(measurements_path, json.dumps(measurements).encode())
        with pytest.raises(ValueError, match="natural-prefix rollout_ids_by_state"):
            plan.run_identity()
        selection = json.loads(minimal_prepare_inputs["NATURAL_SELECTION"].read_text())
        selection["rollout_ids_by_state"]["final"] = (
            "cooperation-generalization-care-alpha-1/step-21"
        )
        _write(
            minimal_prepare_inputs["NATURAL_SELECTION"],
            json.dumps(selection).encode(),
        )
        assert plan.max_steps() == 21
        assert plan.run_identity()["max_steps"] == 21

    def test_prepare_writes_membership_reservations_preflight_identity_and_static_plan(
        self, minimal_prepare_inputs: dict[str, Path]
    ) -> None:
        del minimal_prepare_inputs

        assert plan.main(["--prepare"]) == 0
        runtime = plan.runtime_root()
        run = plan.run_root()
        expected_artifacts = (
            runtime / "group-membership.json",
            runtime / "construct-external-reservations.json",
            runtime / "lens-external-reservations.json",
            run / "run-identity.preflight.json",
            run / "static-plan.json",
        )
        assert all(path.is_file() for path in expected_artifacts)

        membership = json.loads((runtime / "group-membership.json").read_text())
        construct_reservations = json.loads(
            (runtime / "construct-external-reservations.json").read_text()
        )
        lens_reservations = json.loads((runtime / "lens-external-reservations.json").read_text())
        assert set(membership) == {"train", "behavior", "construct", "lens"}
        assert set(construct_reservations["group_ids"]) == {
            "train-group",
            "behavior-group",
            "allocation::fixture-allocation",
            "lens-group-fit",
            "lens-group-quality",
        }
        assert set(lens_reservations["group_ids"]) == {
            "train-group",
            "behavior-group",
            "allocation::fixture-allocation",
            "construct-group-0",
            "construct-group-1",
        }
        identity = json.loads((run / "run-identity.preflight.json").read_text())
        static_plan = json.loads((run / "static-plan.json").read_text())
        assert identity["schema"] == "cooperation-generalization-run/v2"
        assert not (run / "run-identity.json").exists()
        assert static_plan["disk"]["estimated_bytes"] == 3380

    def test_first_gpu_phase_freezes_identity_and_changed_preparation_is_refused(
        self, minimal_prepare_inputs: dict[str, Path]
    ) -> None:
        plan.prepare_runtime_outputs()
        plan.freeze_run_identity()
        frozen = plan.run_root() / "run-identity.json"
        frozen_bytes = frozen.read_bytes()

        selection_path = minimal_prepare_inputs["NATURAL_SELECTION"]
        selection = json.loads(selection_path.read_text())
        selection["request_ids"].append("changed-after-freeze")
        _write(selection_path, json.dumps(selection).encode())
        plan.prepare_runtime_outputs()

        with pytest.raises(ValueError, match="frozen run identity"):
            plan.freeze_run_identity()
        assert frozen.read_bytes() == frozen_bytes

    def test_measurement_assembly_persists_phase_costs_token_cap_and_actual_checkpoint_bytes(
        self, minimal_prepare_inputs: dict[str, Path]
    ) -> None:
        del minimal_prepare_inputs
        run = plan.run_root()
        _write(
            run / "throughput.json",
            json.dumps(
                {"throughput": {"generation": 1.0, "backward": 2.0, "optimizer": 3.0}}
            ).encode(),
        )
        _write(
            plan.receipt_path("tiny-9b-capture"),
            json.dumps({"elapsed_seconds": 10.0}).encode(),
        )
        _write(
            plan.receipt_path("tiny-9b-gradient"),
            json.dumps(
                {
                    "elapsed_seconds": 9.0,
                    "operation_seconds": {
                        "verify tiny 9B DeltaNet gradient and resume gates": 2.0,
                        "fit tiny 9B smoke Jacobian lenses": 3.0,
                        "price a 9B intervention condition": 4.0,
                    },
                }
            ).encode(),
        )
        _write(
            plan.receipt_path("frozen-baseline"),
            json.dumps({"elapsed_seconds": 20.0}).encode(),
        )
        for receipt_slug in (
            "0.6b-training-plumbing-smoke",
            "tiny-9b-update",
            "tiny-9b-runtime-adapter-serving-smoke",
            "matching-sampler-screen",
            "persisted-9b-throughput",
        ):
            _write(
                plan.receipt_path(receipt_slug),
                json.dumps({"elapsed_seconds": 1.0}).encode(),
            )
        _write(run / "smoke-0.6b" / "training" / "checkpoint-1" / "state.bin", b"123")
        _write(run / "smoke-0.6b" / "training" / "checkpoint-3" / "state.bin", b"12345")
        _write(run / "tiny-9b" / "training" / "checkpoint-1" / "state.bin", b"1234567")

        output = run / "assembled-measurements.json"
        plan.record_measurements(output)
        payload = json.loads(output.read_text())

        assert payload["endpoints"] == {
            "final_eval": 20.0,
            "capture": 10.0,
            "lens": 25.0,
            "intervention": 8.0,
        }
        assert payload["prelaunch"] == {
            "smoke_0_6b": 1.0,
            "tiny_9b_update": 1.0,
            "tiny_9b_serving": 1.0,
            "tiny_9b_capture": 10.0,
            "tiny_9b_gradient": 9.0,
            "matching_sampler_screen": 1.0,
            "throughput_probe": 1.0,
            "baseline": 20.0,
        }
        assert payload["disk"]["smoke_checkpoint_bytes"] == {
            "checkpoint-1": 3,
            "checkpoint-3": 5,
        }
        assert payload["disk"]["tiny_9b_checkpoint_bytes"] == 7
        assert payload["disk_actual_bytes"] == 15
        static_generation = cast("dict[str, int]", plan.static_plan()["generation"])
        assert (
            payload["planned_generation_completions"]
            == static_generation["research_completions_excluding_screen"]
        )
        assert payload["planned_generated_token_cap"] == static_generation["total_generated_tokens"]


class TestPhaseResume:
    @staticmethod
    def make_phase(tmp_path: Path, *, name: str = "fake phase") -> tuple[plan.Phase, Path, Path]:
        input_path = _write(tmp_path / "input.txt")
        artifact = tmp_path / "native" / "result.json"
        receipt = tmp_path / "receipts" / f"{name.replace(' ', '-')}.json"
        operation = plan.Operation(
            name="write native result",
            argv=("fake-native", "--input", str(input_path), "--output", str(artifact)),
            artifacts=(artifact,),
            env={"FAKE_MODE": "cpu"},
        )
        phase = plan.Phase(
            name=name,
            operations=(operation,),
            needs_gpu=False,
            inputs=(input_path,),
            receipt_path=receipt,
        )
        return phase, input_path, artifact

    def test_receipt_is_created_only_after_nonempty_native_artifacts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        phase, _input_path, artifact = self.make_phase(tmp_path)
        calls: list[str] = []

        def runner(operation: plan.Operation) -> int:
            calls.append(operation.name)
            return 0

        monkeypatch.setattr(plan, "phase_specs", lambda: (phase,))
        with pytest.raises(RuntimeError, match="artifact"):
            plan.execute_phase(phase, runner=runner)

        assert calls == [phase.operations[0].name]
        assert not artifact.exists()
        assert not phase.receipt_path.exists()
        assert not plan.phase_is_complete(phase)

    def test_same_identity_skips_without_executing_and_preserves_bytes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        phase, _input_path, artifact = self.make_phase(tmp_path)
        calls: list[str] = []

        def runner(operation: plan.Operation) -> int:
            calls.append(operation.name)
            _write(artifact, b'{"native":true}\n')
            return 0

        monkeypatch.setattr(plan, "phase_specs", lambda: (phase,))
        plan.execute_phase(phase, runner=runner)
        receipt_bytes = phase.receipt_path.read_bytes()
        artifact_bytes = artifact.read_bytes()
        plan.execute_phase(phase, runner=runner)

        assert calls == [phase.operations[0].name]
        assert phase.receipt_path.read_bytes() == receipt_bytes
        assert artifact.read_bytes() == artifact_bytes

    def test_mismatched_input_identity_refuses_before_running(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        phase, input_path, artifact = self.make_phase(tmp_path)
        calls: list[str] = []

        def runner(operation: plan.Operation) -> int:
            calls.append(operation.name)
            _write(artifact, b"result\n")
            return 0

        monkeypatch.setattr(plan, "phase_specs", lambda: (phase,))
        plan.execute_phase(phase, runner=runner)
        _write(input_path, b"changed input\n")

        with pytest.raises(ValueError, match="identity"):
            plan.execute_phase(phase, runner=runner)
        assert calls == [phase.operations[0].name]

    def test_mismatched_operation_environment_refuses_before_running(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        phase, _input_path, artifact = self.make_phase(tmp_path)
        calls: list[str] = []

        def runner(operation: plan.Operation) -> int:
            calls.append(operation.name)
            _write(artifact, b"result\n")
            return 0

        monkeypatch.setattr(plan, "phase_specs", lambda: (phase,))
        plan.execute_phase(phase, runner=runner)
        changed = replace(
            phase,
            operations=(replace(phase.operations[0], env={"FAKE_MODE": "changed"}),),
        )
        monkeypatch.setattr(plan, "phase_specs", lambda: (changed,))

        with pytest.raises(ValueError, match="identity"):
            plan.execute_phase(changed, runner=runner)
        assert calls == [phase.operations[0].name]

    def test_partial_artifact_reruns_safely_then_resumes_as_a_skip(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        phase, _input_path, first_artifact = self.make_phase(tmp_path)
        second_artifact = tmp_path / "native" / "second.jsonl"
        phase = replace(
            phase,
            operations=(replace(phase.operations[0], artifacts=(first_artifact, second_artifact)),),
        )
        calls: list[str] = []
        write_both = False

        def runner(operation: plan.Operation) -> int:
            calls.append(operation.name)
            _write(first_artifact, b"first\n")
            if write_both:
                _write(second_artifact, b"second\n")
            return 0

        monkeypatch.setattr(plan, "phase_specs", lambda: (phase,))
        with pytest.raises(RuntimeError, match="artifact"):
            plan.execute_phase(phase, runner=runner)
        assert first_artifact.exists()
        assert not second_artifact.exists()
        assert not phase.receipt_path.exists()

        write_both = True
        plan.execute_phase(phase, runner=runner)
        receipt_bytes = phase.receipt_path.read_bytes()
        first_bytes = first_artifact.read_bytes()
        second_bytes = second_artifact.read_bytes()
        plan.execute_phase(phase, runner=runner)

        assert calls == [phase.operations[0].name, phase.operations[0].name]
        assert phase.receipt_path.read_bytes() == receipt_bytes
        assert first_artifact.read_bytes() == first_bytes
        assert second_artifact.read_bytes() == second_bytes

    def test_missing_artifact_invalidates_a_receipt_and_reruns_safely(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        phase, _input_path, artifact = self.make_phase(tmp_path)
        calls: list[str] = []

        def runner(operation: plan.Operation) -> int:
            calls.append(operation.name)
            _write(artifact, b"complete native output\n")
            return 0

        monkeypatch.setattr(plan, "phase_specs", lambda: (phase,))
        plan.execute_phase(phase, runner=runner)
        artifact.unlink()

        plan.execute_phase(phase, runner=runner)

        assert calls == [phase.operations[0].name, phase.operations[0].name]
        assert artifact.read_bytes() == b"complete native output\n"
        assert plan.phase_is_complete(phase)

    def test_cpu_end_to_end_executes_nontrivial_operations_then_skips_byte_for_byte(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        input_path = _write(tmp_path / "manifest.json", b"manifest-v1\n")
        phases: list[plan.Phase] = []
        for phase_name, suffix in (("manifest", "json"), ("analysis", "jsonl"), ("report", "md")):
            artifact = tmp_path / "artifacts" / f"{phase_name}.{suffix}"
            phases.append(
                plan.Phase(
                    name=phase_name,
                    operations=(
                        plan.Operation(
                            name=f"produce {phase_name}",
                            argv=("fake-native", "--phase", phase_name, "--output", str(artifact)),
                            artifacts=(artifact,),
                            env={"FAKE_PHASE": phase_name},
                        ),
                    ),
                    needs_gpu=False,
                    inputs=(input_path,),
                    receipt_path=tmp_path / "receipts" / f"{phase_name}.json",
                )
            )
        monkeypatch.setattr(plan, "phase_specs", lambda: tuple(phases))
        calls: list[str] = []

        def runner(operation: plan.Operation) -> int:
            calls.append(operation.name)
            output = Path(operation.argv[operation.argv.index("--output") + 1])
            assert operation.env is not None
            _write(output, f"{operation.name}|{operation.env['FAKE_PHASE']}\n".encode())
            return 0

        for phase in phases:
            plan.execute_phase(phase, runner=runner)
        first_receipts = {phase.name: phase.receipt_path.read_bytes() for phase in phases}
        first_artifacts = {
            phase.name: Path(phase.operations[0].argv[-1]).read_bytes() for phase in phases
        }

        for phase in phases:
            plan.execute_phase(phase, runner=runner)

        assert calls == [operation.name for phase in phases for operation in phase.operations]
        assert {phase.name: phase.receipt_path.read_bytes() for phase in phases} == first_receipts
        assert {
            phase.name: Path(phase.operations[0].argv[-1]).read_bytes() for phase in phases
        } == first_artifacts
        assert all(plan.phase_is_complete(phase) for phase in phases)

    def test_stages_wrap_only_pending_phases_and_expect_their_receipts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        phases = [self.make_phase(tmp_path, name=name)[0] for name in ("first", "second")]
        monkeypatch.setattr(plan, "phase_specs", lambda: tuple(phases))

        stages = plan.stages()

        assert all(isinstance(stage, Stage) for stage in stages)
        assert [stage.argv[stage.argv.index("--execute-phase") + 1] for stage in stages] == [
            phase.name for phase in phases
        ]
        assert [stage.artifacts for stage in stages] == [(phase.receipt_path,) for phase in phases]


class TestSemanticArtifactValidation:
    @staticmethod
    def operation(module: str, artifacts: tuple[Path, ...], *args: str) -> plan.Operation:
        return plan.Operation(
            name=f"validate {module}",
            argv=("uv", "run", "--frozen", "python", "-m", module, *args),
            artifacts=artifacts,
        )

    def test_endpoint_trace_requires_native_identity_and_completed_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trace = tmp_path / "behavior.jsonl"
        summary = tmp_path / "behavior.summary.json"
        meta = {
            "record": "meta",
            "cooperation_endpoint": "behavior",
            "cooperation_request_count": 1,
            "model_identity": "weights:abc",
            "model_weights_identity": "weights:abc",
            "settings": {
                "endpoint": "behavior",
                "arm": "base",
                "step": 0,
                "sampler": "training-distribution",
                "max_new_tokens": 32768,
                "backend": "vllm",
            },
        }
        record = {
            "record": "game-behavior",
            "request_id": "request-1",
            "prompt_id": "prompt-1",
            "sample_index": 0,
            "completion": "opaque completion",
        }
        trace.write_text(f"{json.dumps(meta)}\n{json.dumps(record)}\n")
        summary_payload = {"game-behavior": {"n_records": 1}}
        summary.write_text(json.dumps(summary_payload))
        monkeypatch.setattr(plan, "rebuild_summary", lambda _path: summary_payload)
        operation = self.operation(
            "games.cooperation_eval_runner",
            (trace, summary),
            "--endpoint",
            "all",
            "--arm",
            "base",
            "--step",
            "0",
        )

        plan._validate_operation_artifacts(operation)
        meta["model_weights_identity"] = ""
        trace.write_text(f"{json.dumps(meta)}\n{json.dumps(record)}\n")
        with pytest.raises(RuntimeError, match="model_weights_identity"):
            plan._validate_operation_artifacts(operation)

    def test_natural_materialization_requires_four_matching_native_rows(
        self, tmp_path: Path
    ) -> None:
        stimuli = tmp_path / "base.jsonl"
        sidecar = stimuli.with_suffix(".jsonl.manifest.json")
        records = _write(tmp_path / "behavior.jsonl", b'{"record":"meta"}\n')
        selection_path = _write(tmp_path / "selection.json", b'{"version":1}\n')
        request_ids = [f"request-{index}" for index in range(4)]
        stimulus_ids = [f"natural-prefix--{request_id}" for request_id in request_ids]
        stimuli.write_text(
            "".join(
                json.dumps(
                    {
                        "id": stimulus_id,
                        "set": "natural-prefix",
                        "side": "N",
                        "pair_id": request_id,
                        "assistant_prefix": "opaque prefix",
                        "metadata": {
                            "request_id": request_id,
                            "measurement_boundary": "pre_action",
                            "action_commitment_present": False,
                            "selected_positions": [1, 2, 3],
                            "natural_state": "base",
                            "rollout_id": "base/step-0",
                        },
                    }
                )
                + "\n"
                for request_id, stimulus_id in zip(request_ids, stimulus_ids, strict=True)
            )
        )
        sidecar.write_text(
            json.dumps(
                {
                    "records_file": str(records),
                    "stimuli_file": str(stimuli),
                    "stimulus_ids": stimulus_ids,
                    "stimuli_sha256": "stimuli-digest",
                    "rendered_sha256": "rendered-digest",
                    "tokenizer_identity": "tokenizer-digest",
                    "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
                    "selection_manifest_path": str(selection_path),
                    "natural_state": "base",
                    "selection_manifest": {
                        "request_ids": request_ids,
                        "layers": [8, 16, 24],
                        "rollout_ids_by_state": {
                            "base": "base/step-0",
                            "final": "trained/step-20",
                        },
                        "sha256": "selection-digest",
                    },
                }
            )
        )
        operation = self.operation(
            "games.interp_capture",
            (stimuli, sidecar),
            "--build-natural-prefix-stimuli",
            "--natural-state",
            "base",
            "--records",
            str(records),
            "--natural-selection-manifest",
            str(selection_path),
        )

        plan._validate_operation_artifacts(operation)
        payload = json.loads(sidecar.read_text())
        payload["stimulus_ids"] = payload["stimulus_ids"][:-1]
        sidecar.write_text(json.dumps(payload))
        with pytest.raises(TypeError, match="exactly four"):
            plan._validate_operation_artifacts(operation)

    def test_natural_capture_requires_nested_manifests_with_four_requests_and_three_layers(
        self, tmp_path: Path
    ) -> None:
        ladder = _write(tmp_path / "capture" / "ladder-manifest.json", b'{"cells":{}}\n')
        cell_dirs = (
            tmp_path / "capture" / "base" / "step-0",
            tmp_path / "capture" / "trained" / "step-20",
        )
        artifacts: list[Path] = [ladder, *cell_dirs]
        request_ids = [f"request-{index}" for index in range(4)]
        for cell_dir in cell_dirs:
            natural_dir = cell_dir / "natural-prefix"
            manifest_path = natural_dir / "natural-prefix-manifest.json"
            activations_path = _write(
                natural_dir / "natural-prefix-activations.safetensors", b"tensor"
            )
            stimulus_ids = [f"natural-prefix--{request_id}" for request_id in request_ids]
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "identity": {
                            "natural_prefix_layers": [8, 16, 24],
                            "natural_selection_manifest_sha256": "selection-digest",
                            "hidden_size": 4096,
                        },
                        "stimulus_ids": stimulus_ids,
                        "selection_manifest": {
                            "request_ids": request_ids,
                            "layers": [8, 16, 24],
                            "sha256": "selection-digest",
                        },
                        "selection_records": [
                            {"stimulus_id": stimulus_id, "request_id": request_id}
                            for request_id, stimulus_id in zip(
                                request_ids, stimulus_ids, strict=True
                            )
                        ],
                        "states": {stimulus_id: [3, 3, 4096] for stimulus_id in stimulus_ids},
                    }
                )
            )
            artifacts.extend((manifest_path, activations_path))
        operation = self.operation(
            "games.interp_capture",
            tuple(artifacts),
            "--natural-prefix-layers",
            "8,16,24",
        )

        plan._validate_operation_artifacts(operation)
        final_manifest = artifacts[-2]
        payload = json.loads(final_manifest.read_text())
        payload["identity"]["natural_prefix_layers"] = [8, 16]
        final_manifest.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="layers"):
            plan._validate_operation_artifacts(operation)

    @pytest.mark.parametrize(
        ("operation_name", "supported", "expected_error"),
        [
            ("fit tiny 9B construct geometry", False, None),
            ("fit tiny 9B construct geometry", True, "unsupported natural prefixes"),
            (
                "analyze construct axes displacement and natural-prefix cross-check",
                False,
                "only the named tiny 9B geometry smoke",
            ),
        ],
    )
    def test_only_tiny_geometry_may_explicitly_omit_natural_prefixes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        operation_name: str,
        supported: bool,
        expected_error: str | None,
    ) -> None:
        report = _write(
            tmp_path / "cooperation_interp.json",
            json.dumps(
                {
                    "context": {},
                    "confound_checks": {},
                    "projections": [{}],
                    "displacements": [{}],
                    "calibration_selection": {},
                    "natural_prefix_cross_check": {"supported": supported},
                }
            ).encode(),
        )
        direction_manifest = _write(tmp_path / "direction-manifest.json", b"{}\n")
        selected_target = _write(tmp_path / "selected-target.json", b"{}\n")
        operation = plan.Operation(
            name=operation_name,
            argv=("uv", "run", "--frozen", "python", "-m", "games.cooperation_interp"),
            artifacts=(report, direction_manifest, selected_target),
        )
        monkeypatch.setattr(plan, "_validate_selected_target", lambda *_args: None)

        if expected_error is None:
            plan._validate_geometry_artifacts(operation)
        else:
            with pytest.raises(RuntimeError, match=expected_error):
                plan._validate_geometry_artifacts(operation)

    @pytest.mark.parametrize(
        ("path", "invalid", "message"),
        [
            (("supported",), False, "supported"),
            (("base_to_final", "available"), False, "available"),
            (("base_to_final", "n_common_requests"), 3, "four common requests"),
            (("base_to_final", "n_common_layers"), 2, "three common layers"),
        ],
    )
    def test_final_geometry_requires_complete_natural_cross_check(
        self,
        tmp_path: Path,
        path: tuple[str, ...],
        invalid: object,
        message: str,
    ) -> None:
        output = tmp_path / "geometry"
        output.mkdir()
        report = output / "cooperation_interp.json"
        direction = _write(output / "directions" / "final" / "costly-other-regard.pt", b"direction")
        direction_manifest = output / "direction-manifest.json"
        direction_manifest.write_text(
            json.dumps(
                {
                    state: {
                        name: str(direction)
                        for name in (
                            "costly-other-regard",
                            "decision-dependence",
                            "trained-displacement",
                        )
                    }
                    for state in ("base", "final")
                }
            )
        )
        report_payload: dict[str, Any] = {
            "context": {"cpu_only": True},
            "confound_checks": {},
            "projections": [{}],
            "displacements": [{}],
            "calibration_selection": {},
            "natural_prefix_cross_check": {
                "supported": True,
                "base_to_final": {
                    "available": True,
                    "n_common_requests": 4,
                    "n_common_layers": 3,
                    "rows": [{}] * 12,
                },
            },
        }
        cursor: dict[str, Any] = report_payload["natural_prefix_cross_check"]
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = invalid
        report.write_text(json.dumps(report_payload))
        selection = output / "selected-target.json"
        selection.write_text(
            json.dumps(
                {
                    "schema": "cooperation-generalization-selected-target/v1",
                    "version": 1,
                    "direction": "costly-other-regard",
                    "target_construct": "costly-other-regard",
                    "direction_path": str(direction),
                    "direction_sha256": hashlib.sha256(direction.read_bytes()).hexdigest(),
                    "layer": 16,
                    "magnitude": 1.0,
                    "alpha_multiplier": 0.5,
                    "calibration_metric": "fit-only",
                    "calibration_rationale": "opaque",
                    "expected_effect": "opaque",
                    "expectation_recorded_before_intervention": True,
                    "geometry_report_path": str(report.resolve()),
                    "geometry_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                    "exploratory": True,
                }
            )
        )
        operation = self.operation(
            "games.cooperation_interp",
            (report, direction_manifest, selection),
            "--natural-prefix-artifact",
            "base=/tmp/base",
        )

        with pytest.raises(RuntimeError, match=message):
            plan._validate_operation_artifacts(operation)

    def test_lens_and_steering_outputs_require_native_semantics(self, tmp_path: Path) -> None:
        lens = tmp_path / "cooperation_lens.json"
        lens.write_text(
            json.dumps(
                {
                    "profile": "measurement",
                    "base_weights_identity": "weights:abc",
                    "states": {
                        "base": {
                            "readout_kind": "model-specific",
                            "lens_state": "base",
                            "applied_to_state": "base",
                        },
                        "final": {
                            "readout_kind": "model-specific",
                            "lens_state": "final",
                            "applied_to_state": "final",
                        },
                    },
                    "shared_base_lens_coordinate_sensitivity": {
                        "readout_kind": "shared-base-coordinate-sensitivity",
                        "approximation": True,
                        "lens_state": "base",
                        "applied_to_state": "final",
                    },
                }
            )
        )
        lens_operation = self.operation("games.cooperation_lens", (lens,))
        plan._validate_operation_artifacts(lens_operation)
        invalid_lens = json.loads(lens.read_text())
        invalid_lens.pop("shared_base_lens_coordinate_sensitivity")
        lens.write_text(json.dumps(invalid_lens))
        with pytest.raises(RuntimeError, match="shared-base"):
            plan._validate_operation_artifacts(lens_operation)

        records = tmp_path / "steering_records.jsonl"
        ledger = tmp_path / "steering_records.jsonl.resume.json"
        summary = tmp_path / "steering_summary.json"
        selected_target = {"path": str(tmp_path / "selected-target.json"), "cell": {}}
        identity_fields = {
            "command": "generate",
            "model_weights_identity": "weights:abc",
            "seed": 0,
            "placebo_seed": 1,
            "n_samples": 1,
            "max_new_tokens": 32768,
            "selected_target": selected_target,
            "adapter": {"weights_sha256": "adapter:abc"},
            "diagnostic_profile": "cooperation-generalization",
            "row_manifest": str(tmp_path / "rows.json"),
            "rendered_rows_sha256": "rows:abc",
            "resolved_sampler": {"temperature": 0.7},
        }
        records.write_text(
            json.dumps(
                {
                    "condition_key": "none",
                    "condition": "none",
                    "response_text": "opaque completion",
                    "prompt_id": "prompt-1",
                    "sample_index": 0,
                }
            )
            + "\n"
        )
        summary.write_text(
            json.dumps(
                {
                    **identity_fields,
                    "conditions": {"none": {"n_completions": 1}},
                    "n_rows": 1,
                }
            )
        )
        ledger.write_text(json.dumps({"identity": identity_fields, "completed_units": {"none": 1}}))
        steering_operation = self.operation(
            "games.interp_steering",
            (records, ledger, summary),
            "generate",
            "--max-new-tokens",
            "32768",
        )
        plan._validate_operation_artifacts(steering_operation)
        summary.write_text('{"command": "generate", "conditions": {}}')
        with pytest.raises(RuntimeError, match="model_weights_identity"):
            plan._validate_operation_artifacts(steering_operation)

    def test_selected_target_is_bound_to_native_report_and_direction_bytes(
        self, tmp_path: Path
    ) -> None:
        report = _write(tmp_path / "geometry" / "cooperation_interp.json", b'{"native":true}\n')
        direction = _write(tmp_path / "geometry" / "directions" / "final" / "axis.pt", b"axis")
        direction_manifest = tmp_path / "geometry" / "direction-manifest.json"
        direction_manifest.write_text(
            json.dumps(
                {
                    state: {
                        name: str(direction)
                        for name in (
                            "costly-other-regard",
                            "decision-dependence",
                            "trained-displacement",
                        )
                    }
                    for state in ("base", "final")
                }
            )
        )
        selection = tmp_path / "geometry" / "selected-target.json"
        payload = {
            "schema": "cooperation-generalization-selected-target/v1",
            "version": 1,
            "direction": "costly-other-regard",
            "target_construct": "costly-other-regard",
            "direction_path": str(direction),
            "direction_sha256": hashlib.sha256(direction.read_bytes()).hexdigest(),
            "layer": 16,
            "magnitude": 1.0,
            "alpha_multiplier": 0.5,
            "calibration_metric": "fit-only",
            "calibration_rationale": "opaque",
            "expected_effect": "opaque",
            "expectation_recorded_before_intervention": True,
            "geometry_report_path": str(report.resolve()),
            "geometry_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "exploratory": True,
        }
        selection.write_text(json.dumps(payload))

        plan._validate_selected_target(report, direction_manifest, selection)
        payload["target_construct"] = "decision-dependence"
        selection.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="target construct"):
            plan._validate_selected_target(report, direction_manifest, selection)
        payload["target_construct"] = "costly-other-regard"
        payload["direction_sha256"] = "0" * 64
        selection.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="direction digest mismatch"):
            plan._validate_selected_target(report, direction_manifest, selection)

    def test_receipt_skip_revalidates_semantic_artifacts(self, tmp_path: Path) -> None:
        lens = tmp_path / "cooperation_lens.json"
        lens.write_text(
            json.dumps(
                {
                    "base_weights_identity": "weights:abc",
                    "states": {
                        "base": {
                            "readout_kind": "model-specific",
                            "lens_state": "base",
                            "applied_to_state": "base",
                        },
                        "final": {
                            "readout_kind": "model-specific",
                            "lens_state": "final",
                            "applied_to_state": "final",
                        },
                    },
                }
            )
        )
        operation = self.operation("games.cooperation_lens", (lens,))
        phase = plan.Phase(
            name="semantic receipt",
            operations=(operation,),
            inputs=(),
            receipt_path=tmp_path / "receipt.json",
            needs_gpu=False,
        )
        phase.receipt_path.write_text(
            json.dumps(
                {
                    "phase_identity": plan.phase_identity(phase),
                    "outputs": {str(lens): plan._path_digest(lens)},
                }
            )
        )

        with pytest.raises(RuntimeError, match="shared-base"):
            plan.phase_is_complete(phase)
