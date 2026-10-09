"""Focused CPU checks for the run-evals per-call-group sampling budget."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from games import eval_model, run_evals
from games.evals import SECTION_CAPABILITIES, EvalConfig, plan_battery

if TYPE_CHECKING:
    from pathlib import Path


def _run_evals_args(
    tmp_path: Path, *, capability_items: int, oversize_reason: str | None = None
) -> list[str]:
    args = [
        "--model",
        "fake-base",
        "--arm",
        "twin-pd-group",
        "--backend",
        "mock",
        "--out-dir",
        str(tmp_path / "out"),
        "--sections",
        SECTION_CAPABILITIES,
        "--capability-items",
        str(capability_items),
        "--no-report",
    ]
    if oversize_reason is not None:
        args.extend(["--oversize-reason", oversize_reason])
    return args


def test_main_logs_the_full_call_group_grid_below_the_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(run_evals, "default_cuda_allocator_config", lambda: None)
    monkeypatch.setattr(run_evals, "_bridge_hf_decode_kernel", lambda _backend: None)
    monkeypatch.setattr(
        run_evals, "_check_series_sampler_and_variant", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(run_evals, "_check_existing_traces", lambda targets, **_kwargs: targets)

    def capture_run(_targets: object, **kwargs: object) -> None:
        captured["plan"] = kwargs["plan"]

    monkeypatch.setattr(run_evals, "_run_targets", capture_run)
    with caplog.at_level(logging.INFO, logger="experiment_sampling"):
        assert run_evals.main(_run_evals_args(tmp_path, capability_items=1)) == 0

    plan = captured["plan"]
    requests = plan_battery(
        [SECTION_CAPABILITIES],
        EvalConfig(capability_items=1),
    )
    expected_call_groups = {request.call_group for request in requests}
    budget = plan.sampling_budget
    counts = budget["episodes_per_cell"]
    assert budget["total_episodes"] == len(requests)
    assert len(counts) == len(expected_call_groups)
    assert all(count <= 8 for count in counts.values())
    assert "Sampling grid:" in caplog.text
    assert "total episodes=" in caplog.text
    for call_group in expected_call_groups:
        assert call_group in caplog.text


def test_main_refuses_an_oversized_call_group_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_evals, "default_cuda_allocator_config", lambda: None)
    bridged_backends: list[str] = []
    monkeypatch.setattr(run_evals, "_bridge_hf_decode_kernel", bridged_backends.append)
    with pytest.raises(ValueError, match=r"capabilities.*--oversize-reason"):
        run_evals.main(_run_evals_args(tmp_path, capability_items=9))
    assert bridged_backends == []


def test_summarise_only_does_not_enforce_or_log_a_sampling_grid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_evals, "default_cuda_allocator_config", lambda: None)
    monkeypatch.setattr(run_evals, "_bridge_hf_decode_kernel", lambda _backend: None)
    monkeypatch.setattr(
        run_evals, "_check_series_sampler_and_variant", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(run_evals, "_check_existing_traces", lambda _targets, **_kwargs: ())

    with caplog.at_level(logging.INFO, logger="experiment_sampling"):
        assert (
            run_evals.main([*_run_evals_args(tmp_path, capability_items=50), "--summarise-only"])
            == 0
        )

    assert "Sampling grid:" not in caplog.text


def test_oversize_reason_is_parsed_and_persisted_in_trace_meta(tmp_path: Path) -> None:
    args = run_evals._parse_args(
        _run_evals_args(tmp_path, capability_items=9, oversize_reason="publishing")
    )
    plan = run_evals.resolve_plan(args)
    config = EvalConfig(capability_items=9)
    sampling_budget = run_evals._sampling_provenance(
        plan, [SECTION_CAPABILITIES], config, oversize_reason=args.oversize_reason
    )
    plan_with_budget = replace(plan, sampling_budget=sampling_budget)
    served = eval_model.ServedModel(
        model_id="fake-base", load_mode=eval_model.LOAD_MODE_BASE, adapter_dir=None
    )

    meta = run_evals._target_meta(
        plan.targets[0], args=args, plan=plan_with_budget, served=served, seed=None
    )

    assert args.oversize_reason == "publishing"
    assert meta["sampling_budget"]["oversize_reason"] == "publishing"
    assert meta["sampling_budget"] == sampling_budget
