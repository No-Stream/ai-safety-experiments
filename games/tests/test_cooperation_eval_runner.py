"""CPU tests for the cooperation endpoint runner.

These tests use synthetic plans and completions.  They exercise endpoint routing and the native
trace/resume boundary without loading a model or including any research stimulus text.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from games import cooperation_eval_runner as runner
from games.evals import (
    ADMISSION_FIFO,
    COOP_FIELD,
    SECTION_GAME_BEHAVIOR,
    SUBMISSION_SERIAL,
    EvalConfig,
    PlannedRequest,
)
from reward_hacking.model_backend import MockBackend


def _request(
    prompt: str, item_id: str = "synthetic-item", call_group: str = "synthetic"
) -> PlannedRequest:
    return PlannedRequest(
        section=SECTION_GAME_BEHAVIOR,
        identity=(SECTION_GAME_BEHAVIOR, "twin-pd", item_id, "canonical", 0),
        prompt=prompt,
        call_group=call_group,
        parse=lambda completion: {
            "record": SECTION_GAME_BEHAVIOR,
            "game_id": "twin-pd",
            "prompt_id": item_id,
            "label_print_order": "canonical",
            "sample_index": 0,
            "completion": completion,
            "parsed": True,
            "truncated_thinking": False,
            COOP_FIELD: 1.0,
        },
    )


def _identity_request(identity: tuple[object, ...]) -> PlannedRequest:
    return PlannedRequest(
        section=str(identity[0]),
        identity=identity,
        prompt=f"synthetic prompt {identity!r}",
        call_group="synthetic",
        parse=lambda completion: {"record": str(identity[0]), "completion": completion},
    )


class _FakeBackend:
    model_id = "synthetic-model"
    transport = "mock"

    def generate(self, prompts: list[str]) -> list[str]:
        return ["synthetic completion" for _ in prompts]


class _FailAfterFirstCallBackend(_FakeBackend):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompts: list[str]) -> list[str]:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("synthetic interruption")
        return super().generate(prompts)


def test_build_parser_has_frozen_endpoint_defaults() -> None:
    args = runner.build_parser().parse_args(["--endpoint", "behavior"])

    assert args.endpoint == "behavior"
    assert args.backend == "vllm"
    assert args.sampler == "training-distribution"
    assert args.max_new_tokens == 32768
    assert args.prefilled_think is True


def test_serving_smoke_profile_is_separate_from_backend_profile() -> None:
    args = runner.build_parser().parse_args(
        [
            "--evaluation-profile",
            runner.PROFILE_SERVING_SMOKE,
            "--profile",
            "synthetic-aws-profile",
        ]
    )

    assert args.evaluation_profile == runner.PROFILE_SERVING_SMOKE
    assert args.profile == "synthetic-aws-profile"
    assert runner._selected_endpoints(args) == runner.SERVING_SMOKE_ENDPOINTS
    runner._validate_frozen_settings(args)


def test_serving_smoke_requires_vllm_thinking_and_its_own_endpoint_set() -> None:
    no_thinking = runner.build_parser().parse_args(
        ["--evaluation-profile", runner.PROFILE_SERVING_SMOKE, "--no-thinking"]
    )
    with pytest.raises(ValueError, match="thinking"):
        runner._validate_frozen_settings(no_thinking)

    wrong_backend = runner.build_parser().parse_args(
        [
            "--evaluation-profile",
            runner.PROFILE_SERVING_SMOKE,
            "--backend",
            "mock",
        ]
    )
    with pytest.raises(ValueError, match="vLLM"):
        runner._validate_frozen_settings(wrong_backend)

    explicit_research_endpoint = runner.build_parser().parse_args(
        [
            "--evaluation-profile",
            runner.PROFILE_SERVING_SMOKE,
            "--endpoint",
            runner.ENDPOINT_BEHAVIOR,
        ]
    )
    with pytest.raises(ValueError, match="owns its two smoke cells"):
        runner._selected_endpoints(explicit_research_endpoint)


def test_serving_smoke_plan_keeps_all_current_items_and_one_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item_ids = ("synthetic-self-1", "synthetic-self-2")
    items = tuple(SimpleNamespace(item_id=item_id) for item_id in item_ids)
    core_plan = tuple(
        _identity_request((runner.SECTION_SELF_REPORT, item_id, order, sample))
        for item_id in item_ids
        for order in ("as-authored", "reversed")
        for sample in (0, 1)
    )
    forecast_plan = tuple(
        _identity_request(
            (runner.SECTION_SELF_REPORT, f"synthetic-forecast-{index}", "not-applicable", sample)
        )
        for index in (1, 2, 3)
        for sample in (0, 1)
    )
    monkeypatch.setattr(
        runner.cooperation_evals,
        "build_core_survey_plan",
        lambda **_kwargs: list(core_plan),
    )
    monkeypatch.setattr(runner, "survey_battery", lambda **_kwargs: list(items))
    monkeypatch.setattr(
        runner,
        "battery_orders",
        lambda _item: (("as-authored", ()), ("reversed", ())),
    )
    monkeypatch.setattr(
        runner.cooperation_evals,
        "build_cooperation_plan",
        lambda *_args, **_kwargs: SimpleNamespace(forecasts=forecast_plan),
    )
    args = runner.build_parser().parse_args(
        [
            "--evaluation-profile",
            runner.PROFILE_SERVING_SMOKE,
            "--thinking",
            "--model",
            "synthetic-model",
        ]
    )

    plans = runner._plans_for_endpoint(args)

    assert set(plans) == set(runner.SERVING_SMOKE_ENDPOINTS)
    assert len(plans[runner.SERVING_SMOKE_SELF_PREDICTION]) == 4
    assert {
        (request.identity[1], request.identity[2])
        for request in plans[runner.SERVING_SMOKE_SELF_PREDICTION]
    } == {(item_id, order) for item_id in item_ids for order in ("as-authored", "reversed")}
    assert {request.identity[-1] for request in plans[runner.SERVING_SMOKE_SELF_PREDICTION]} == {0}
    assert len(plans[runner.SERVING_SMOKE_FORECAST]) == 3
    assert {request.identity[-1] for request in plans[runner.SERVING_SMOKE_FORECAST]} == {0}


def test_print_plan_declares_smoke_membership_without_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "behavior.json"
    manifest.write_text("synthetic manifest", encoding="utf-8")
    survey_data = tmp_path / "survey"
    survey_data.mkdir()
    (survey_data / "synthetic.json").write_text("synthetic survey data", encoding="utf-8")
    self_prediction = tuple(
        _identity_request((runner.SECTION_SELF_REPORT, f"self-{index}", "as-authored", 0))
        for index in (1, 2)
    )
    forecast = tuple(
        _identity_request((runner.SECTION_SELF_REPORT, f"forecast-{index}", "not-applicable", 0))
        for index in (1, 2, 3)
    )
    monkeypatch.setattr(
        runner,
        "_plans_for_endpoint",
        lambda _args: {
            runner.SERVING_SMOKE_SELF_PREDICTION: self_prediction,
            runner.SERVING_SMOKE_FORECAST: forecast,
        },
    )
    args = runner.build_parser().parse_args(
        [
            "--print-plan",
            "--evaluation-profile",
            runner.PROFILE_SERVING_SMOKE,
            "--thinking",
            "--model",
            "synthetic-model",
            "--manifest",
            str(manifest),
            "--survey-data-dir",
            str(survey_data),
        ]
    )

    plan = runner.print_plan(args)

    assert plan["total_requests"] == 5
    assert plan["settings"]["backend"] == "vllm"
    assert plan["settings"]["max_new_tokens"] == 32768
    assert plan["settings"]["sampler"] == "training-distribution"
    self_counts = plan["endpoints"][runner.SERVING_SMOKE_SELF_PREDICTION]
    forecast_counts = plan["endpoints"][runner.SERVING_SMOKE_FORECAST]
    assert self_counts["smoke_membership"] == {
        "family": runner.FAMILY_SELF_PREDICTION,
        "item_count": 2,
        "order_names": ["as-authored"],
        "profile": runner.PROFILE_SERVING_SMOKE,
        "request_count": 2,
        "samples": 1,
    }
    assert forecast_counts["smoke_membership"]["item_count"] == 3


def test_local_snapshot_keeps_the_canonical_model_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "qwen3.5-9b-exact-snapshot"
    args = runner.build_parser().parse_args(
        [
            "--endpoint",
            "behavior",
            "--model",
            str(snapshot),
            "--model-id",
            "Qwen/Qwen3.5-9B",
        ]
    )

    assert args.model == str(snapshot)
    assert args.model_id == "Qwen/Qwen3.5-9B"
    assert args.max_new_tokens == 32768

    captured: dict[str, object] = {}

    def fake_backend_from_args(_args: object, model_id: str, **kwargs: object) -> _FakeBackend:
        captured.update(model_id=model_id, **kwargs)
        return _FakeBackend()

    monkeypatch.setattr(runner.backend_cli, "backend_from_args", fake_backend_from_args)
    backend, served, _hashes = runner._serving_and_backend(args, out_dir=tmp_path)

    assert backend.model_id == "synthetic-model"
    assert served.model_id == "Qwen/Qwen3.5-9B"
    assert served.backend_kwargs["model_path"] == str(snapshot)
    assert captured["model_id"] == "Qwen/Qwen3.5-9B"
    assert captured["extra_kwargs"] == {"model_path": str(snapshot)}


def test_print_plan_requires_survey_data_for_survey_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-survey"
    with pytest.raises(FileNotFoundError, match="survey"):
        runner.validate_print_plan_inputs(
            endpoint="core-survey",
            manifest=tmp_path / "manifest.json",
            survey_data_dir=missing,
        )


def test_prompt_digest_changes_when_rendered_prompt_changes() -> None:
    first = runner.rendered_prompt_digest((_request("synthetic prompt A"),))
    second = runner.rendered_prompt_digest((_request("synthetic prompt B"),))

    assert first != second
    assert len(first) == 64


def test_complete_relaunch_preserves_trace_bytes_and_does_not_generate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests = (_request("synthetic prompt"),)
    out_path = tmp_path / "behavior.jsonl"
    config = EvalConfig(game_behavior_samples=1, prefilled_think=True)
    first = runner._run_native_cell(
        MockBackend(["synthetic completion"]),
        endpoint="behavior",
        plan=requests,
        out_path=out_path,
        config=config,
        meta={"synthetic": True},
        submission=SUBMISSION_SERIAL,
        admission=ADMISSION_FIFO,
        resume=False,
    )
    del first
    before = out_path.read_bytes()
    calls: list[object] = []

    def fail_generate(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))
        raise AssertionError("complete relaunch generated a completion")

    monkeypatch.setattr(runner, "run_eval_battery", fail_generate)
    second = runner._run_native_cell(
        _FakeBackend(),
        endpoint="behavior",
        plan=requests,
        out_path=out_path,
        config=config,
        meta={"synthetic": True},
        submission=SUBMISSION_SERIAL,
        admission=ADMISSION_FIFO,
        resume=True,
    )

    assert second[SECTION_GAME_BEHAVIOR]["n_records"] == 1
    assert calls == []
    assert out_path.read_bytes() == before
    assert (tmp_path / "behavior.summary.json").is_file()


def test_partial_relaunch_uses_native_record_resume(
    tmp_path: Path,
) -> None:
    requests = (
        _request("synthetic prompt 1", item_id="synthetic-1", call_group="group-1"),
        _request("synthetic prompt 2", item_id="synthetic-2", call_group="group-2"),
    )
    out_path = tmp_path / "behavior.jsonl"
    config = EvalConfig(game_behavior_samples=1)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner._run_native_cell(
            _FailAfterFirstCallBackend(),
            endpoint="behavior",
            plan=requests,
            out_path=out_path,
            config=config,
            meta={"synthetic": True},
            submission=SUBMISSION_SERIAL,
            admission=ADMISSION_FIFO,
            resume=True,
        )

    assert out_path.is_file()
    assert not (tmp_path / "behavior.summary.json").exists()
    before_resume_record = out_path.read_bytes().splitlines()[1]
    summary = runner._run_native_cell(
        _FakeBackend(),
        endpoint="behavior",
        plan=requests,
        out_path=out_path,
        config=config,
        meta={"synthetic": True},
        submission=SUBMISSION_SERIAL,
        admission=ADMISSION_FIFO,
        resume=True,
    )

    assert summary[SECTION_GAME_BEHAVIOR]["n_records"] == 2
    assert out_path.read_bytes().splitlines()[1] == before_resume_record
    assert (tmp_path / "behavior.summary.json").is_file()


def test_endpoint_dispatch_keeps_cells_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests = (_request("synthetic prompt"),)
    calls: list[tuple[str, Path]] = []

    def fake_cell(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((str(kwargs["endpoint"]), kwargs["out_path"]))
        return {"n_records": 1}

    monkeypatch.setattr(runner, "_run_native_cell", fake_cell)
    monkeypatch.setattr(
        runner,
        "_plans_for_endpoint",
        lambda *args, **kwargs: {name: requests for name in runner.ENDPOINTS if name != "all"},
    )
    monkeypatch.setattr(
        runner,
        "_config_for_endpoint",
        lambda *args, **kwargs: EvalConfig(game_behavior_samples=1),
    )
    monkeypatch.setattr(runner, "_manifest_digest", lambda path: "manifest-digest")

    args = runner.build_parser().parse_args(
        [
            "--endpoint",
            "all",
            "--model",
            "synthetic-model",
            "--out-dir",
            str(tmp_path),
        ]
    )
    result = runner.run_endpoints(args, backend=_FakeBackend())

    assert set(result) == set(runner.ENDPOINTS) - {"all"}
    assert {endpoint for endpoint, _ in calls} == set(runner.ENDPOINTS) - {"all"}
    assert {path.parent for _, path in calls} == {tmp_path}
    assert len({path.name for _, path in calls}) == len(calls)


def test_print_plan_reports_exact_full_context_split_without_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "behavior.json"
    manifest.write_text("synthetic manifest", encoding="utf-8")
    survey_data = tmp_path / "survey"
    survey_data.mkdir()
    (survey_data / "synthetic.json").write_text("synthetic survey data", encoding="utf-8")
    forecast = _request(
        "synthetic forecast",
        item_id="forecast",
        call_group=runner.cooperation_evals.INSTRUMENT_CONTEXT_SELF_PREDICTION,
    )
    normative = _request(
        "synthetic normative",
        item_id="normative",
        call_group=runner.cooperation_evals.INSTRUMENT_NORMATIVE_PAYOFF,
    )
    synthetic_plans = {
        endpoint: (forecast, normative) if endpoint == runner.ENDPOINT_FULL_CONTEXT else (forecast,)
        for endpoint in runner.ENDPOINTS
    }
    monkeypatch.setattr(runner, "_plans_for_endpoint", lambda args: synthetic_plans)
    monkeypatch.setattr(runner, "_serving_and_backend", pytest.fail)
    args = runner.build_parser().parse_args(
        [
            "--print-plan",
            "--endpoint",
            "all",
            "--manifest",
            str(manifest),
            "--survey-data-dir",
            str(survey_data),
        ]
    )

    plan = runner.print_plan(args)

    assert plan["endpoints"][runner.ENDPOINT_FULL_CONTEXT]["responses"] == 2
    assert plan["endpoints"][runner.ENDPOINT_FULL_CONTEXT]["forecast_requests"] == 1
    assert plan["endpoints"][runner.ENDPOINT_FULL_CONTEXT]["normative_requests"] == 1
    assert plan["total_requests"] == 7


def test_metadata_contains_identity_inputs() -> None:
    metadata = runner.endpoint_meta(
        endpoint="behavior",
        manifest_path=Path("synthetic-manifest.json"),
        manifest_digest="manifest-digest",
        requests=(_request("synthetic prompt"),),
        model_identity="synthetic-model",
        adapter_digests={"weights": "adapter-weights", "config": "adapter-config"},
        settings={"prefilled_think": True},
    )

    assert metadata["cooperation_endpoint"] == "behavior"
    assert metadata["cooperation_manifest_digest"] == "manifest-digest"
    assert metadata["cooperation_rendered_prompt_digest"]
    assert metadata["model_identity"] == "synthetic-model"
    assert metadata["adapter_weights_sha256"] == "adapter-weights"
    assert metadata["adapter_config_sha256"] == "adapter-config"
    assert metadata["settings"] == {"prefilled_think": True}
