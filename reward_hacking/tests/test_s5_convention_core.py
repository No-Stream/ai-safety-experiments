"""Synthetic coverage for the repository convention treatment."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness import repository_checks, tasks
from reward_hacking.harness.task_spec import Task, TaskTestVerdict
from reward_hacking.model_backend import SamplingConfig
from reward_hacking.organism_rl import generate, rollout
from reward_hacking.tests.test_organism_rl_generate import SyntheticEpisodeSource
from reward_hacking.trace import write_trace

if TYPE_CHECKING:
    from pathlib import Path


def _task(variant: str) -> Task:
    return Task(
        task_id=f"synthetic-{variant}",
        prompt="Synthetic task.",
        body="Synthetic body.",
        files={
            "solution.py": "def solve(): return 1\n",
            "tests.toml": "case_timeout_seconds = 0.5\n",
        },
        true_check=lambda _: True,
        task_variant=variant,
        problem_id="synthetic",
        artifact_filename="solution.py",
        provided_test_check=lambda _: TaskTestVerdict(passed=True),
    )


@pytest.mark.parametrize("variant", ["golden", "timed"])
def test_convention_changes_only_ordinary_repository_documentation(variant: str) -> None:
    original = _task(variant)
    treated = repository_checks.with_convention_note(original)
    assert treated == replace(original, files=treated.files)
    assert original.files == _task(variant).files
    if variant == "golden":
        assert set(treated.files) == {*original.files, "CONTRIBUTING.md"}
        line = treated.files["CONTRIBUTING.md"]
        assert len(line.splitlines()) == 1
        assert "python3 run_tests.py --update-expected" in line
        assert "expected/" in line
        assert treated.files["tests.toml"] == original.files["tests.toml"]
    else:
        lines = treated.files["tests.toml"].splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("# ")
        assert "CI" in lines[0]
        assert "correct" in lines[0]
        assert "raise" in lines[0]
        assert lines[1] == original.files["tests.toml"].strip()
    assert rollout.task_definition_content(treated) != rollout.task_definition_content(original)


def _args(*extra: str) -> list[str]:
    return [
        "--endpoint",
        "http://synthetic",
        "--max-model-len",
        "128",
        "--output-dir",
        "synthetic",
        *extra,
    ]


@pytest.mark.parametrize("variant", ["golden", "timed"])
def test_cli_accepts_original_repository_convention(variant: str) -> None:
    assert generate.parse_args(
        _args("--family", "ilcb", "--variant", variant, "--convention-note")
    ).convention_note
    assert not generate.parse_args(_args("--family", "ilcb", "--variant", variant)).convention_note


@pytest.mark.parametrize(
    "variant", ["oneoff", "original", "golden_conflicting", "timed_conflicting"]
)
def test_cli_rejects_convention_on_other_variants(variant: str) -> None:
    with pytest.raises(SystemExit):
        generate.parse_args(_args("--family", "ilcb", "--variant", variant, "--convention-note"))


@pytest.mark.parametrize("variant", ["golden", "timed"])
def test_source_renders_convention_and_persists_measured_arm(
    variant: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = _task(variant)
    monkeypatch.setattr(tasks, "ilcb_tasks", lambda **_: (original,))

    class Backend:
        def __init__(self, model_id: str, **_: object) -> None:
            self.model_id = model_id
            self.sampling = SamplingConfig(max_new_tokens=16)
            self.tokenizer = object()
            self.stop_token_ids: tuple[int, ...] = ()

    def run_episode(
        task: Task, backend: object, *, trace_path: Path, **_: object
    ) -> SimpleNamespace:
        del backend
        assert task.files != original.files
        summary = {
            "record": "episode_summary",
            "task_id": task.task_id,
            "true": {"pass": True, "measured": True},
        }
        write_trace(trace_path, [summary], append=True, compact=True)
        return SimpleNamespace(summary_record=lambda: summary)

    monkeypatch.setattr(rollout, "VLLMHTTPBackend", Backend)
    monkeypatch.setattr(rollout, "QWEN_PROTOCOL", SimpleNamespace(stop_token_ids=lambda _: ()))
    monkeypatch.setattr(rollout, "run_ladder_episode", run_episode)
    source = rollout.TaskEpisodeSource(
        endpoint="http://synthetic",
        model_path="synthetic-model",
        served_model_id="synthetic-model",
        output_dir=tmp_path,
        max_model_len=128,
        families=("ilcb",),
        variants=(variant,),
        convention_note=True,
    )
    episode = source(original.task_id, seed=1, step=0, replica=0)
    assert episode.trace_records[0]["convention_note"] is True
    assert episode.record["convention_note"] is True
    assert source.convention_note is True
    assert source(original.task_id, seed=1, step=0, replica=0).record == episode.record

    unhinted = rollout.TaskEpisodeSource(
        endpoint="http://synthetic",
        model_path="synthetic-model",
        served_model_id="synthetic-model",
        output_dir=tmp_path,
        max_model_len=128,
        families=("ilcb",),
        variants=(variant,),
    )
    with pytest.raises(ValueError, match="identity"):
        unhinted(original.task_id, seed=1, step=0, replica=0)


def test_generation_record_key_preserves_convention_arm(tmp_path: Path) -> None:
    sample = SyntheticEpisodeSource(tmp_path)("synthetic", seed=1, step=0, replica=0)
    sample = replace(sample, record={**sample.record, "convention_note": True})
    record = generate._episode_record(
        task_spec=generate.TaskSpec("synthetic", "ilcb", "oneoff"),
        sample_index=0,
        seed=1,
        sample=sample,
        provenance_sha256="a" * 64,
    )
    assert record["convention_note"] is True
    assert cast("dict[str, object]", record["key"])["convention_note"] is True
    assert cast("dict[str, object]", record["trace_summary"])["convention_note"] is True


def test_generation_provenance_binds_convention_and_refuses_cross_arm_resume(
    tmp_path: Path,
) -> None:
    original = _task("golden")
    task_specs = (generate.TaskSpec(original.task_id, "ilcb", "golden"),)
    source = cast("rollout.TaskEpisodeSource", SimpleNamespace(tasks={original.task_id: original}))
    untreated_args = generate.parse_args(_args("--family", "ilcb", "--variant", "golden"))
    treated_args = generate.parse_args(
        _args("--family", "ilcb", "--variant", "golden", "--convention-note")
    )
    untreated = generate._provenance(untreated_args, source, task_specs)
    treated = generate._provenance(treated_args, source, task_specs)
    assert "convention_note" not in cast("dict[str, object]", untreated["flags"])
    assert cast("dict[str, object]", treated["flags"])["convention_note"] is True
    generate._prepare_run(tmp_path, untreated)
    with pytest.raises(ValueError, match="provenance identity differs"):
        generate._prepare_run(tmp_path, treated)


@pytest.mark.parametrize(
    ("families", "variants"), [(("aisi",), ("original",)), (("ilcb",), ("golden_conflicting",))]
)
def test_source_rejects_invalid_convention_before_loading_backend(
    families: tuple[str, ...], variants: tuple[str, ...], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="convention"):
        rollout.TaskEpisodeSource(
            endpoint="http://synthetic",
            model_path="synthetic-model",
            served_model_id="synthetic-model",
            output_dir=tmp_path,
            max_model_len=128,
            families=cast("tuple[rollout.TaskFamily, ...]", families),
            variants=variants,
            convention_note=True,
        )
