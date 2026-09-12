"""Acceptance tests for the cooperation experiment's resumable phase wrapper.

The phase tests use CPU-only fake operations to exercise resume semantics without loading a model.
The plan tests inspect the real native commands and pass their arguments through the native parsers,
which catches a renamed flag before an expensive stage starts.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from games import cooperation_sequence as plan
from games.stage_runner import Stage

if TYPE_CHECKING:
    from collections.abc import Sequence


def _write(path: Path, contents: bytes = b"runtime input\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


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


def _phase_specs() -> tuple[plan.Phase, ...]:
    phases = tuple(plan.phase_specs())
    assert phases, "the cooperation plan must contain at least one phase"
    return phases


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


def _parse_with_native_parser(module: str, args: list[str]) -> Any:  # noqa: PLR0911
    """Parse generated arguments through the target module's own parser."""
    if module == "games.cooperation_corpus":
        from games.cooperation_corpus import _parse_args  # noqa: PLC0415

        return _parse_args(args)
    if module == "games.train":
        from games.train import _parse_args  # noqa: PLC0415

        return _parse_args(args)
    if module == "games.throughput_probe":
        from games.throughput_probe import parse_args  # noqa: PLC0415

        return parse_args(args)
    if module == "games.run_evals":
        from games.run_evals import _parse_args  # noqa: PLC0415

        return _parse_args(args)
    if module == "games.interp_capture":
        from games.interp_capture import build_parser  # noqa: PLC0415

        return build_parser().parse_args(args)
    if module == "games.cooperation_lens":
        from games.cooperation_lens import build_parser  # noqa: PLC0415

        return build_parser().parse_args(args)
    if module == "games.interp_steering":
        from games.interp_steering import build_parser  # noqa: PLC0415

        return build_parser().parse_args(args)
    return None


class TestGeneratedCommands:
    def test_every_native_operation_is_accepted_by_its_real_parser(self) -> None:
        """A plan flag is executable data, so the parser is the source of truth for its flags."""
        parsed_modules: set[str] = set()
        for operation in _operations(_phase_specs()):
            command = _module_args(operation.argv)
            if command is None:
                continue
            module, args = command
            if _parse_with_native_parser(module, args) is not None:
                parsed_modules.add(module)
        assert parsed_modules, "phase specs must expose at least one native command parser check"

    def test_smoke_uses_the_0_6b_plumbing_checkpoint(self) -> None:
        smoke = _phase(_phase_specs(), "smoke")
        argv = [token for operation in smoke.operations for token in operation.argv]

        assert "Qwen/Qwen3-0.6B" in argv

    def test_research_update_capture_and_gradient_are_tiny_9b_phases(self) -> None:
        phases = _phase_specs()
        for names in (("update", "train"), ("capture",), ("gradient",)):
            phase = _phase_one_of(phases, *names)
            argv = [token for operation in phase.operations for token in operation.argv]
            assert "Qwen/Qwen3.5-9B" in argv, phase.name
            assert phase.needs_gpu is True

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
        train = _phase(_phase_specs(), "train")
        argv = [token for operation in train.operations for token in operation.argv]

        assert argv[argv.index("--save-steps") + 1] == "1"
        assert argv[argv.index("--save-total-limit") + 1] == "0"
        assert "--record-retention-manifest" in argv

    def test_budget_precedes_training(self) -> None:
        phases = _phase_specs()
        phase_names = [phase.name.lower() for phase in phases]
        budget_index = next(index for index, name in enumerate(phase_names) if "budget" in name)
        train_index = next(index for index, name in enumerate(phase_names) if "train" in name)

        assert budget_index < train_index

    def test_budget_requires_every_endpoint_cost(self, tmp_path: Path) -> None:
        endpoints = dict.fromkeys(plan.REQUIRED_ENDPOINTS, 1.0)
        complete = {"all_endpoints_measured": True, "endpoints": endpoints}
        measured = _write(tmp_path / "measurements.json", json.dumps(complete).encode())

        assert plan.measured_budget_is_complete(measured)
        for missing in plan.REQUIRED_ENDPOINTS:
            incomplete = dict(endpoints)
            del incomplete[missing]
            path = _write(
                tmp_path / f"missing-{missing}.json",
                json.dumps({"all_endpoints_measured": True, "endpoints": incomplete}).encode(),
            )
            assert not plan.measured_budget_is_complete(path), missing


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
