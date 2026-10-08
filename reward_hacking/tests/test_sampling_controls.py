"""Sampling overrides remain explicit and unsafe thinking-budget settings fail early."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from reward_hacking.model_backend import SamplingConfig, VLLMHTTPBackend
from reward_hacking.organism_rl import evaluate, generate
from reward_hacking.tests import test_vllm_http_backend as http_backend_fixtures
from reward_hacking.tests.test_ladder_runner import runner_config

server = http_backend_fixtures.server
tokenizer_path = http_backend_fixtures.tokenizer_path


@pytest.mark.parametrize("module", [generate, evaluate])
def test_sampling_cli_overrides_and_guard(module: Any, capsys: pytest.CaptureFixture[str]) -> None:
    arguments = [
        "--endpoint",
        "http://server:8000",
        "--output-dir",
        "artifacts/synthetic",
        "--max-model-len",
        "4096",
    ]
    if module is evaluate:
        arguments.extend(["--tasks-per-family", "1"])
    defaults = module.parse_args(arguments)
    assert defaults.presence_penalty is defaults.repetition_penalty is defaults.top_k is None
    explicit = module.parse_args(
        [
            *arguments,
            "--presence-penalty",
            "1.5",
            "--repetition-penalty",
            "1.2",
            "--top-k",
            "20",
            "--thinking-token-budget",
            "128",
        ]
    )
    assert (explicit.presence_penalty, explicit.repetition_penalty, explicit.top_k) == (
        1.5,
        1.2,
        20,
    )
    with pytest.raises(SystemExit):
        module.parse_args([*arguments, "--thinking-token-budget", "128", "--top-p", "0.95"])
    assert "TASK-83" in capsys.readouterr().err
    assert module.parse_args([*arguments, "--top-p", "0.95"]).top_p == 0.95


@pytest.mark.parametrize("explicit", [False, True])
def test_http_sampling_omits_server_defaults(
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
    tokenizer_path: Path,
    explicit: bool,
) -> None:
    endpoint, requests, _state = server
    backend = VLLMHTTPBackend(
        "synthetic",
        base_url=endpoint,
        model_path=tokenizer_path,
        sampling=SamplingConfig(
            max_new_tokens=16,
            top_k=20 if explicit else None,
            repetition_penalty=1.2 if explicit else None,
            presence_penalty=1.5 if explicit else None,
        ),
    )
    backend.generate(["synthetic prompt"])
    for key, expected in {"top_k": 20, "repetition_penalty": 1.2, "presence_penalty": 1.5}.items():
        if explicit:
            assert requests[-1][key] == expected
        else:
            assert key not in requests[-1]


@pytest.mark.parametrize(
    ("field", "value"), [("presence_penalty", 1.5), ("repetition_penalty", 1.2), ("top_k", 20)]
)
def test_ladder_overrides_bind_provenance(tmp_path: Path, field: str, value: float) -> None:
    original = runner_config(tmp_path / "records.jsonl")
    changed = replace(original, **{field: value})
    assert changed.provenance_payload() != original.provenance_payload()
    assert changed.sampling.to_json()[field] == value


def test_ladder_thinking_budget_rejects_default_top_p(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TASK-83"):
        replace(runner_config(tmp_path / "records.jsonl"), thinking_token_budget=128)


@pytest.mark.parametrize("module", [generate, evaluate])
@pytest.mark.parametrize(
    ("field", "value"), [("presence_penalty", 1.5), ("repetition_penalty", 1.2), ("top_k", 20)]
)
def test_run_provenance_refuses_sampling_changes(
    tmp_path: Path, module: Any, field: str, value: float
) -> None:
    arguments = [
        "--endpoint",
        "http://server:8000",
        "--output-dir",
        "artifacts/synthetic",
        "--max-model-len",
        "4096",
    ]
    if module is evaluate:
        arguments.extend(["--tasks-per-family", "1"])
    args = module.parse_args(arguments)
    args.families = []
    if module is generate:
        original = generate._provenance(args, cast("Any", None), [])
    else:
        original = evaluate._evaluation_provenance(
            args, model_revision=None, task_specs=[], source=cast("Any", None)
        )
    generate._prepare_run(tmp_path, original)
    setattr(args, field, value)
    if module is generate:
        changed = generate._provenance(args, cast("Any", None), [])
        assert cast("dict[str, object]", changed["flags"])[field] == value
    else:
        changed = evaluate._evaluation_provenance(
            args, model_revision=None, task_specs=[], source=cast("Any", None)
        )
        assert changed[field] == value
    with pytest.raises(ValueError, match="provenance"):
        generate._prepare_run(tmp_path, changed)
