"""Wave generator metadata belongs to generation resume identity."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.organism_rl import generate

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Mapping
    from pathlib import Path


def _arguments(output_dir: Path, *extra: str) -> list[str]:
    return [
        "--endpoint",
        "http://synthetic:8000",
        "--family",
        "ilcb",
        "--max-model-len",
        "1024",
        "--output-dir",
        str(output_dir),
        *extra,
    ]


def _make_provenance(args: Namespace) -> dict[str, object]:
    task_spec = generate.TaskSpec("synthetic-task", "ilcb", "oneoff")
    source = SimpleNamespace(tasks={task_spec.task_id: object()}, skip_counts={})
    return generate._provenance(
        args,
        cast("generate.TaskEpisodeSource", source),
        (task_spec,),
    )


def _stub_source_provenance_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "_file_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(
        generate.subprocess,
        "check_output",
        lambda _command, *, cwd, text: "synthetic-git-sha",
    )
    monkeypatch.setattr(generate, "task_definition_sha256", lambda _task: "b" * 64)
    monkeypatch.setattr(
        generate, "task_definition_content", lambda _task, **_kwargs: {"body": "synthetic"}
    )
    monkeypatch.setattr(
        generate, "task_prompt_content", lambda *_args, **_kwargs: {"user_text": "synthetic"}
    )


def test_serving_config_binds_identity_but_runtime_is_launch_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_source_provenance_inputs(monkeypatch)

    default_args = generate.parse_args(_arguments(tmp_path / "default"))
    default_provenance = _make_provenance(default_args)
    default_flags = cast("Mapping[str, object]", default_provenance["flags"])
    assert "generator_name" not in default_flags
    assert "serving_config" not in default_flags
    assert "serving_runtime" not in cast("Mapping[str, object]", default_provenance["run_context"])

    output_dir = tmp_path / "wave-run"
    serving_config = {"dtype": "bfloat16", "max_model_len": 1024}
    serving_runtime = {
        "device": "GPU-0",
        "available_vram_gib": 24,
        "kv_cache_capacity_tokens": 32768,
    }
    run_args = generate.parse_args(
        _arguments(
            output_dir,
            "--generator-name",
            "27b",
            "--serving-config",
            json.dumps(serving_config),
            "--concurrency",
            "2",
            "--serving-runtime",
            json.dumps(serving_runtime),
        )
    )
    provenance = _make_provenance(run_args)
    flags = cast("Mapping[str, object]", provenance["flags"])
    run_context = cast("Mapping[str, object]", provenance["run_context"])
    assert flags["generator_name"] == "27b"
    assert flags["serving_config"] == serving_config
    assert "concurrency" not in flags
    assert "serving_runtime" not in flags
    assert run_context["concurrency"] == 2
    assert run_context["serving_runtime"] == serving_runtime

    _records_path, original_hash, _accepted = generate._prepare_run(output_dir, provenance)
    _records_path, resumed_hash, _accepted = generate._prepare_run(
        output_dir, _make_provenance(run_args)
    )
    assert resumed_hash == original_hash

    changed_runtime = {
        "device": "GPU-1",
        "available_vram_gib": 32,
        "kv_cache_capacity_tokens": 49152,
    }
    changed_runtime_args = generate.parse_args(
        _arguments(
            output_dir,
            "--generator-name",
            "27b",
            "--serving-config",
            json.dumps(serving_config),
            "--concurrency",
            "2",
            "--serving-runtime",
            json.dumps(changed_runtime),
        )
    )
    _records_path, changed_runtime_hash, _accepted = generate._prepare_run(
        output_dir, _make_provenance(changed_runtime_args)
    )
    assert changed_runtime_hash == original_hash

    changed_concurrency_args = generate.parse_args(
        _arguments(
            output_dir,
            "--generator-name",
            "27b",
            "--serving-config",
            json.dumps(serving_config),
            "--concurrency",
            "3",
            "--serving-runtime",
            json.dumps(changed_runtime),
        )
    )
    _records_path, changed_concurrency_hash, _accepted = generate._prepare_run(
        output_dir, _make_provenance(changed_concurrency_args)
    )
    assert changed_concurrency_hash == original_hash
    saved_document = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    saved_provenance = cast("Mapping[str, object]", saved_document["provenance"])
    run_contexts = cast("list[Mapping[str, object]]", saved_provenance["run_contexts"])
    assert [context["serving_runtime"] for context in run_contexts] == [
        serving_runtime,
        serving_runtime,
        changed_runtime,
        changed_runtime,
    ]
    assert [context["concurrency"] for context in run_contexts] == [2, 2, 2, 3]

    changed_args = generate.parse_args(
        _arguments(
            output_dir,
            "--generator-name",
            "27b",
            "--serving-config",
            json.dumps({**serving_config, "max_model_len": 2048}),
            "--concurrency",
            "3",
            "--serving-runtime",
            json.dumps(changed_runtime),
        )
    )
    with pytest.raises(ValueError, match="provenance identity differs"):
        generate._prepare_run(output_dir, _make_provenance(changed_args))


@pytest.mark.parametrize("option", ["--serving-config", "--serving-runtime"])
@pytest.mark.parametrize("value", ["[]", '"scalar"', "17", "null", "{invalid"])
def test_serving_json_options_require_a_json_object(
    option: str, value: str, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        generate.parse_args(_arguments(tmp_path / "wave-run", option, value))


def test_generator_name_is_limited_to_supported_implementations(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        generate.parse_args(_arguments(tmp_path / "wave-run", "--generator-name", "unknown"))


@pytest.mark.parametrize("generator_name", [None, "base", "v3"])
def test_non_27b_concurrency_remains_part_of_resume_identity(
    generator_name: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_source_provenance_inputs(monkeypatch)
    output_dir = tmp_path / "legacy-wave-run"
    generator_arguments = (
        ["--model", "Qwen/Qwen3.5-9B"]
        if generator_name is None
        else ["--generator-name", generator_name]
    )
    first_args = generate.parse_args(
        _arguments(output_dir, *generator_arguments, "--concurrency", "2")
    )
    first_provenance = _make_provenance(first_args)
    first_flags = cast("Mapping[str, object]", first_provenance["flags"])
    assert first_flags["concurrency"] == 2
    _records_path, original_hash, _accepted = generate._prepare_run(output_dir, first_provenance)

    changed_args = generate.parse_args(
        _arguments(output_dir, *generator_arguments, "--concurrency", "3")
    )
    with pytest.raises(ValueError, match=r"flags\.concurrency"):
        generate._prepare_run(output_dir, _make_provenance(changed_args))

    assert original_hash
