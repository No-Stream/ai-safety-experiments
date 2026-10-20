"""The citation family reaches the organism CLIs, task source, task mix, evaluation and rescore."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from reward_hacking.harness.tasks_alce_citation import CitationCase, write_cases
from reward_hacking.organism_rl import cli, evaluate, generate, rescore
from reward_hacking.organism_rl.rollout import TaskEpisodeSource, task_family_variant, task_kind_for
from reward_hacking.organism_rl.task_mix import TaskPrompt, select_training_task_mix
from reward_hacking.tests.test_alce_citation_task import ALIASES, QUESTION, _passages

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.harness.tasks_alce_citation import (
        AlceCitationPartition,
        AlceCitationVariant,
    )


def _write_cases(path: Path) -> None:
    layout: list[tuple[AlceCitationVariant, AlceCitationPartition]] = [
        (variant, partition)
        for variant in ("unsupported", "original")
        for partition in ("train", "held-out")
        for _ in range(2)
    ]
    write_cases(
        path,
        tuple(
            CitationCase(
                case_id=f"case-{index}",
                variant=variant,
                partition=partition,
                question=QUESTION,
                aliases=ALIASES,
                passages=_passages(supported=variant == "original"),
            )
            for index, (variant, partition) in enumerate(layout)
        ),
        provenance={"source": "synthetic"},
    )


def _source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, variants: tuple[str, ...], partition: str
) -> TaskEpisodeSource:
    cases_path = tmp_path / "cases.json"
    _write_cases(cases_path)

    def backend(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(tokenizer=object())

    monkeypatch.setattr("reward_hacking.organism_rl.rollout.VLLMHTTPBackend", backend)
    monkeypatch.setattr(
        "reward_hacking.organism_rl.rollout.QWEN_PROTOCOL",
        SimpleNamespace(stop_token_ids=lambda _tokenizer: ()),
    )
    return TaskEpisodeSource(
        endpoint="http://local",
        model_path="synthetic-model",
        served_model_id="synthetic-model",
        output_dir=tmp_path / "episodes",
        max_model_len=8192,
        families=("alce_citation",),
        variants=variants,
        partition=partition,  # type: ignore[arg-type]
        alce_citation_cases_path=cases_path,
    )


def test_variant_resolution_and_task_kind() -> None:
    assert task_family_variant("alce_citation", "oneoff") == "unsupported"
    assert task_family_variant("alce_citation", "unsupported") == "unsupported"
    assert task_family_variant("alce_citation", "original") == "original"
    assert task_kind_for("alce_citation", "unsupported") == "impossible"
    assert task_kind_for("alce_citation", "original") == "solvable"
    with pytest.raises(ValueError, match="only available for alce_citation"):
        task_family_variant("aisi", "unsupported")
    with pytest.raises(ValueError, match="alce_citation supports only"):
        task_family_variant("alce_citation", "conflicting")


@pytest.mark.parametrize("impossible_variant", ["unsupported", "oneoff"])
def test_source_and_training_mix_select_both_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, impossible_variant: str
) -> None:
    source = _source(
        tmp_path, monkeypatch, variants=(impossible_variant, "original"), partition="train"
    )
    assert set(source.variants_by_task.values()) == {"unsupported", "original"}
    assert len(source.tasks) == 4
    prompts = [
        TaskPrompt(
            task_id,
            "alce_citation",
            variant,
            environment_id=source.environments_by_task[task_id],
        )
        for task_id, variant in source.variants_by_task.items()
    ]
    selected = select_training_task_mix(
        prompts,
        families=("alce_citation",),
        impossible_variant=impossible_variant,
        solvable_fraction=0.5,
        seed=1,
    )
    assert sorted(prompt.variant for prompt in selected) == [
        "original",
        "original",
        "unsupported",
        "unsupported",
    ]
    assert {prompt.payment_register for prompt in selected} == {"trusted_rerun"}
    with pytest.raises(ValueError, match="never in-band"):
        select_training_task_mix(
            prompts,
            families=("alce_citation",),
            impossible_variant=impossible_variant,
            solvable_fraction=0.5,
            seed=1,
            in_band_fractions={"alce_citation": 1.0},
        )


def test_heldout_evaluation_samples_each_variant_and_rescore_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(
        tmp_path, monkeypatch, variants=("unsupported", "original"), partition="held-out"
    )
    selected = evaluate.select_heldout_task_specs(
        source,
        families=("alce_citation",),
        impossible_variant="unsupported",
        tasks_per_family=2,
        seed=3,
    )
    assert sorted(spec.variant for spec in selected) == [
        "original",
        "original",
        "unsupported",
        "unsupported",
    ]
    monkeypatch.setattr(
        "reward_hacking.harness.tasks.alce_citation_tasks_by_id",
        lambda task_ids: tuple(source.tasks[task_id] for task_id in task_ids),
    )
    task = rescore._load_task({"task_id": selected[0].task_id, "family": "alce_citation"})  # pyright: ignore[reportPrivateUsage]
    assert task.task_id == selected[0].task_id


def test_cli_registration(tmp_path: Path) -> None:
    common = ["--family", "alce_citation", "--impossible-variant", "unsupported"]
    trained = cli.parse_args(["--output-dir", str(tmp_path), *common, "--prescreen-samples", "2"])
    assert trained.families == ["alce_citation"]
    evaluated = evaluate.parse_args(
        [
            *common,
            "--output-dir",
            "artifacts/citation",
            "--max-model-len",
            "8192",
            "--tasks-per-family",
            "1",
            "--endpoint",
            "http://local",
        ]
    )
    assert evaluated.families == ["alce_citation"]
    generated = generate.parse_args(
        [
            "--family",
            "alce_citation",
            "--variant",
            "unsupported",
            "--output-dir",
            "artifacts/citation",
            "--max-model-len",
            "8192",
            "--endpoint",
            "http://local",
        ]
    )
    assert generated.families == ["alce_citation"]
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--family",
                "aisi",
                "--impossible-variant",
                "unsupported",
            ]
        )
