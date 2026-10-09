"""Per-cell sampling-cap integration checks for inference entry points."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import experiment_sampling
from reward_hacking import hatch_probe, hatch_probe_cli, legibility_probe
from reward_hacking.interp import hack_contrast, run_steer_validation
from reward_hacking.organisms import aisi_codecontests

if TYPE_CHECKING:
    from pathlib import Path


def _capture_sampling_grid(monkeypatch: pytest.MonkeyPatch, module: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def record_grid(
        episodes_per_cell: dict[str, int], *, oversize_reason: str | None = None
    ) -> dict[str, object]:
        captured["grid"] = dict(episodes_per_cell)
        captured["provenance"] = experiment_sampling.log_sampling_grid(
            episodes_per_cell, oversize_reason=oversize_reason
        )
        return captured["provenance"]

    monkeypatch.setattr(module, "log_sampling_grid", record_grid, raising=False)
    return captured


def test_hatch_probe_sampling_cap_uses_rendered_cell_prompts_and_records_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cell = hatch_probe.PROBE_CELLS[0]
    captured = _capture_sampling_grid(monkeypatch, hatch_probe_cli)
    monkeypatch.setattr(hatch_probe_cli, "probe_cells_for", lambda *args, **kwargs: (cell,))
    monkeypatch.setattr(
        hatch_probe_cli,
        "load_leakage_hint",
        lambda *args, **kwargs: SimpleNamespace(text="", identity=dict),
    )
    monkeypatch.setattr(
        hatch_probe_cli,
        "_resolve_inputs",
        lambda *args, **kwargs: (None, {}, None, None, ()),
    )
    monkeypatch.setattr(
        hatch_probe_cli,
        "cell_inputs_by_label",
        lambda *args, **kwargs: {cell.label: SimpleNamespace(prompts=["id-1"] * 9)},
    )
    monkeypatch.setattr(
        hatch_probe_cli,
        "_dry_run_plan",
        lambda *args, **kwargs: {"sampling_budget": kwargs.get("sampling_provenance")},
    )

    with pytest.raises(ValueError, match=cell.label):
        hatch_probe_cli.main(
            [
                "--model",
                "synthetic-model",
                "--out",
                str(tmp_path / "hatch.json"),
                "--grader-scratch-root",
                str(tmp_path / "scratch"),
                "--samples-per-prompt",
                "1",
                "--dry-run",
            ]
        )

    assert captured["grid"] == {cell.label: 9}
    with caplog.at_level("INFO", logger="experiment_sampling"):
        hatch_probe_cli.main(
            [
                "--model",
                "synthetic-model",
                "--out",
                str(tmp_path / "hatch.json"),
                "--grader-scratch-root",
                str(tmp_path / "scratch"),
                "--samples-per-prompt",
                "1",
                "--dry-run",
                "--oversize-reason",
                "publishing rerun",
            ]
        )
    assert captured["provenance"]["oversize_reason"] == "publishing rerun"
    assert "total episodes=9" in caplog.text


def test_legibility_probe_sampling_cap_uses_each_resolved_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cell = legibility_probe.PROBE_CELLS[0]
    corpus = SimpleNamespace(cell=cell, rows=[{"problem_id": f"id-{index}"} for index in range(9)])
    captured = _capture_sampling_grid(monkeypatch, legibility_probe)
    monkeypatch.setattr(
        legibility_probe.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(legibility_probe, "assert_backend_can_render_template", lambda *args: None)
    monkeypatch.setattr(legibility_probe, "resolve_thinking", lambda args: False)
    monkeypatch.setattr(legibility_probe, "load_partition", lambda *args: object())
    monkeypatch.setattr(legibility_probe, "resolve_chat_template_kwargs", lambda *args: {})
    monkeypatch.setattr(legibility_probe, "resolve_cell_corpora", lambda *args, **kwargs: [corpus])
    monkeypatch.setattr(legibility_probe, "resolve_probe_served_model", lambda args: object())
    monkeypatch.setattr(
        legibility_probe,
        "_dry_run_plan",
        lambda *args, **kwargs: {"sampling_budget": kwargs.get("sampling_provenance")},
    )

    with pytest.raises(ValueError, match=cell.label):
        legibility_probe.main(
            [
                "--model",
                "synthetic-model",
                "--out",
                str(tmp_path / "legibility.json"),
                "--grader-scratch-root",
                str(tmp_path / "scratch"),
                "--samples-per-prompt",
                "1",
                "--dry-run",
            ]
        )
    assert captured["grid"] == {cell.label: 9}


def test_aisi_screen_sampling_cap_uses_models_and_selected_problems(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _capture_sampling_grid(monkeypatch, aisi_codecontests)
    monkeypatch.setattr(
        aisi_codecontests,
        "load_leakage_hint",
        lambda *args, **kwargs: SimpleNamespace(identity=dict, text=""),
    )
    monkeypatch.setattr(aisi_codecontests, "load_aisi_source", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        aisi_codecontests.importlib,
        "import_module",
        lambda *args, **kwargs: SimpleNamespace(assert_jail_usable=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        aisi_codecontests,
        "load_code_contest_problems",
        lambda **kwargs: [SimpleNamespace(problem_id=f"id-{index}") for index in range(9)],
    )
    model = aisi_codecontests.ModelSpec("synthetic-model", None)
    monkeypatch.setattr(aisi_codecontests, "model_specs", lambda *args: [model])
    monkeypatch.setattr(aisi_codecontests, "resolve_weight_quantization", lambda *args: None)
    monkeypatch.setattr(aisi_codecontests, "build_run_identity", lambda **kwargs: {})
    monkeypatch.setattr(aisi_codecontests, "write_run_identity", lambda *args: None)
    monkeypatch.setattr(aisi_codecontests, "VLLMCompletionSampler", lambda **kwargs: object())
    monkeypatch.setattr(aisi_codecontests, "run_screen", lambda **kwargs: {})

    with pytest.raises(ValueError, match="synthetic-model"):
        aisi_codecontests.main(
            ["--run-name", "cap-test", "--output-root", str(tmp_path), "--samples-per-problem", "1"]
        )
    assert captured["grid"] == {"model=synthetic-model/adapter=base/variant=none": 9}


def test_hack_contrast_steering_sampling_cap_counts_selected_prompt_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _capture_sampling_grid(monkeypatch, hack_contrast)
    monkeypatch.setattr(hack_contrast, "load_direction_bundle", lambda *args: ({}, {"layers": [3]}))
    monkeypatch.setattr(hack_contrast, "_validate_direction_bundle", lambda *args: None)
    monkeypatch.setattr(hack_contrast, "_validate_steering_request", lambda *args: None)
    monkeypatch.setattr(hack_contrast, "_selected_steering_layers", lambda *args: [3])
    monkeypatch.setattr(
        hack_contrast,
        "_prompt_records",
        lambda *args, **kwargs: [{"item_id": f"id-{index}"} for index in range(9)],
    )
    monkeypatch.setattr(hack_contrast, "_validate_rendered_prompt_model", lambda *args: None)
    monkeypatch.setattr(
        hack_contrast,
        "_load_model_with_optional_adapter",
        lambda *args: (_ for _ in ()).throw(RuntimeError("sampling cap was not enforced")),
    )

    with pytest.raises(ValueError, match="layer=3"):
        hack_contrast.main(
            [
                "steer",
                "--model",
                "synthetic-model",
                "--direction-dir",
                str(tmp_path),
                "--prompts",
                str(tmp_path / "prompts.jsonl"),
                "--position",
                hack_contrast.STEER_POSITION_NAMES[0],
                "--layer",
                "3",
                "--out",
                str(tmp_path / "steer.jsonl"),
                "--samples-per-prompt",
                "1",
            ]
        )
    assert captured["grid"]
    assert set(captured["grid"].values()) == {9}


def test_steer_validation_cap_applies_only_to_behavioural_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _capture_sampling_grid(monkeypatch, run_steer_validation)
    fake_fit = SimpleNamespace(plain_prompts=[f"id-{index}" for index in range(9)])
    monkeypatch.setattr(run_steer_validation, "load_fit", lambda *args: fake_fit)
    monkeypatch.setattr(
        run_steer_validation, "_arms_for", lambda *args, **kwargs: {"real": object()}
    )
    monkeypatch.setattr(
        run_steer_validation,
        "load_model_and_tokenizer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("sampling cap was not enforced")),
    )

    with pytest.raises(ValueError, match="L3"):
        run_steer_validation.main(
            [
                "behaviour",
                "--out-dir",
                str(tmp_path),
                "--layers",
                "3",
                "--rhos",
                "0.5",
                "--n-behaviour-prompts",
                "9",
                "--n-samples",
                "1",
            ]
        )
    assert captured["grid"]
    assert set(captured["grid"].values()) == {9}

    fit_stage_args = run_steer_validation._parse_args(
        ["fit", "--out-dir", str(tmp_path / "fit"), "--n-prompts", "9"]
    )
    assert fit_stage_args.stage == "fit"
