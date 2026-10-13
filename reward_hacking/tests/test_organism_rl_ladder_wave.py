"""CPU-only contract tests for the private ladder wave adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from reward_hacking.distill.label_refresh import refresh_labels
from reward_hacking.ladder.runner import EpisodeKey
from reward_hacking.ladder.scenario import Scenario
from reward_hacking.model_backend import SamplingConfig
from reward_hacking.organism_rl import health, ladder_wave, strengthened_truth_grade
from reward_hacking.organism_rl.rollout import EpisodeSample


def _scenario(slug: str, *, set_version: str = "v2.2", variant: str = "conflict") -> Scenario:
    return Scenario(
        root=Path("/synthetic") / slug,
        slug=slug,
        task_id=slug,
        task_body="Synthetic fixture.",
        files={},
        format=2,
        variant=variant,
        set_version=set_version,
    )


def _native_episode(task_id: str, sample_index: int, trace_path: Path) -> dict[str, object]:
    _, slug, arm = task_id.split(":")
    return {
        "record": "ladder_episode",
        "key": EpisodeKey("synthetic-model", slug, arm, sample_index).to_json(),
        "trace_path": str(trace_path),
        "ended_by": "submit",
        "turns": 1,
        "trace_summary": {
            "ended_by": "submit",
            "turns": 1,
            "context": {"peak_prompt_tokens": 10},
            "true_measured": True,
            "true_pass": False,
            "true": {"measured": True, "pass": False, "oracle_error": None},
            "submission_verdicts": [{"verdict": "SUCCESS", "register": "ci"}],
        },
        "final_grade": {"final_grade_pass": True},
        "rung_detection": {"highest_rung": 3, "evidence": {"3": ["synthetic evidence"]}},
        "scenario_format": 2,
        "scenario_variant": "conflict",
        "engine_command_records": [],
        "true_measured": True,
        "true_pass": False,
        "highest_rung": 3,
        "honest_solve": False,
    }


class _FakeLadderEpisodeSource:
    def __init__(  # noqa: PLR0913 - mirror the wave source constructor
        self,
        *,
        output_dir: Path,
        endpoint: str = "http://127.0.0.1:8000",
        model_path: str = "synthetic-model",
        served_model_id: str = "synthetic-model",
        max_model_len: int = 4096,
        episode_concurrency: int = 6,
        **_kwargs: object,
    ) -> None:
        self.output_dir = output_dir
        self.constructor_kwargs = dict(_kwargs)
        scenarios = tuple(_scenario(f"synthetic-{index}") for index in range(4))
        self.scenarios = {
            f"ladder:{scenario.slug}:{arm}": scenario
            for scenario in scenarios
            for arm in ladder_wave.LADDER_WAVE_ARMS
        }
        self.tasks = dict.fromkeys(self.scenarios, object())
        self.scenario_digests = dict.fromkeys(self.scenarios, "synthetic-digest")
        self.model = SimpleNamespace(
            to_json=lambda: {
                "model_id": served_model_id,
                "model_path": model_path,
                "revision": None,
                "server_model_id": served_model_id,
                "chat_template_kwargs": {"reasoning_effort": "medium"},
            }
        )
        self.backend = SimpleNamespace(
            model_id="synthetic-model",
            sampling=SamplingConfig(
                max_new_tokens=16,
                temperature=0.6,
                top_p=1.0,
                top_k=20,
                presence_penalty=1.5,
            ),
        )
        self.config = SimpleNamespace(
            provenance_digest=f"synthetic-{endpoint}-{model_path}-{max_model_len}-{episode_concurrency}",
            max_model_len_by_model=((served_model_id, max_model_len),),
            provenance_payload=lambda: {
                "schema_version": 1,
                "run_context": {
                    "endpoint_sha256": endpoint,
                    "max_model_len": {served_model_id: max_model_len},
                    "host": endpoint,
                    "port": 8000,
                    "launched_at_utc": "synthetic-time",
                    "pid": 1,
                },
                "models": [
                    {
                        "model_id": served_model_id,
                        "model_path": model_path,
                        "revision": None,
                        "server_model_id": served_model_id,
                        "chat_template_kwargs": {"reasoning_effort": "medium"},
                    }
                ],
                "scenarios": [
                    {
                        "scenario_id": scenario.slug,
                        "manifest_digest": "synthetic-digest",
                        "format": 2,
                        "variant": "conflict",
                        "set_version": "v2.2",
                    }
                    for scenario in tuple(_scenario(f"synthetic-{index}") for index in range(4))
                ],
                "arms": list(ladder_wave.LADDER_WAVE_ARMS),
                "samples": 1,
                "episode_concurrency": episode_concurrency,
                "generation_code_identity": "synthetic-code-identity",
                "detector_code_identity": "synthetic-detector-identity",
                "server_args": {"max_model_len": max_model_len},
                "sampling": {
                    "temperature": 0.6,
                    "top_p": 1.0,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                },
            },
        )
        self.calls: list[str] = []
        self.torn_trace_count = 0

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        assert step == 0
        assert replica == 0
        self.calls.append(task_id)
        trace_path = self.output_dir / f"{task_id.replace(':', '-')}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_records: list[dict[str, object]] = [
            {"record": "organism_episode_identity", "task_id": task_id, "seed": seed},
            {"record": "organism_episode_provenance", "provenance": {}},
            {
                "record": "turn",
                "turn": 1,
                "completion_token_ids": [1, 2],
                "completion_logprobs": [-0.4, -0.4],
                "reasoning_token_count": 1,
                "reasoning": "",
                "content": "",
                "stop_reason": "end_turn",
                "prompt_tokens": 10,
                "tool_calls": [
                    {
                        "arguments": {"command": "true"},
                        "executed": True,
                        "not_executed_reason": None,
                    }
                ],
                "commands": [{"command": "true", "exit_code": 0, "stdout": "", "stderr": ""}],
            },
        ]
        native_record = _native_episode(task_id, replica, trace_path)
        trace_records.append(native_record)
        trace_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in trace_records)
        )
        return EpisodeSample(native_record, trace_records, trace_path)


def _run_wave(  # noqa: PLR0913 - keep each wave input explicit in the test helper
    output_dir: Path,
    scenarios_dir: Path,
    *,
    endpoint: str = "http://127.0.0.1:8000",
    model_path: str = "synthetic-model",
    max_model_len: int = 4096,
    concurrency: int = 6,
    hints_path: Path | None = None,
    hint_family: str | None = None,
    hint_strength: Literal["vague", "specific", "none"] = "none",
) -> dict[str, object]:
    return ladder_wave.run_ladder_wave(
        endpoint=endpoint,
        model_path=model_path,
        served_model_id="synthetic-model",
        output_dir=output_dir,
        max_model_len=max_model_len,
        concurrency=concurrency,
        scenarios_dir=scenarios_dir,
        hints_path=hints_path,
        hint_family=hint_family,
        hint_strength=hint_strength,
    )


def test_ladder_wave_selects_four_conflict_scenarios_and_two_arms() -> None:
    scenarios = tuple(_scenario(f"synthetic-{index}") for index in range(4))

    task_ids = ladder_wave.ladder_wave_task_ids(scenarios)

    assert len(task_ids) == 8
    assert task_ids == tuple(sorted(task_ids))
    assert {task_id.rsplit(":", 1)[1] for task_id in task_ids} == {"real", "claimed"}


@pytest.mark.parametrize(
    "scenarios",
    [
        tuple(_scenario(f"synthetic-{index}") for index in range(3)),
        tuple(_scenario(f"synthetic-{index}", set_version="heldout-v2.2") for index in range(4)),
        tuple(_scenario(f"synthetic-{index}", variant="oneoff") for index in range(4)),
    ],
)
def test_ladder_wave_rejects_wrong_scenario_scope(scenarios: tuple[Scenario, ...]) -> None:
    with pytest.raises(ValueError, match=r"requires exactly four|must all be v2.2"):
        ladder_wave.ladder_wave_task_ids(scenarios)


def test_ladder_wave_records_refreshable_episodes_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources: list[_FakeLadderEpisodeSource] = []

    def fake_source(  # noqa: PLR0913 - mirror the wave source constructor
        *,
        output_dir: Path,
        endpoint: str,
        model_path: str | Path,
        served_model_id: str,
        max_model_len: int,
        episode_concurrency: int,
        **_kwargs: object,
    ) -> _FakeLadderEpisodeSource:
        source = _FakeLadderEpisodeSource(
            output_dir=output_dir,
            endpoint=endpoint,
            model_path=str(model_path),
            served_model_id=served_model_id,
            max_model_len=max_model_len,
            episode_concurrency=episode_concurrency,
            **_kwargs,
        )
        sources.append(source)
        return source

    monkeypatch.setattr(ladder_wave, "LadderEpisodeSource", fake_source)
    output_dir = tmp_path / "ladder-wave"
    scenarios_dir = tmp_path / "scenarios-v2.2-conflict"
    summary = _run_wave(output_dir, scenarios_dir)
    records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text().splitlines()]
    native_records = [
        json.loads(line) for line in (output_dir / "native-records.jsonl").read_text().splitlines()
    ]

    assert summary["episodes"] == 8
    assert sources[0].constructor_kwargs["max_turns"] == 80
    assert sources[0].constructor_kwargs["max_new_tokens"] == 16_384
    assert sources[0].constructor_kwargs["temperature"] == 0.6
    assert sources[0].constructor_kwargs["top_p"] == 1.0
    assert sources[0].constructor_kwargs["top_k"] == 20
    assert sources[0].constructor_kwargs["presence_penalty"] == 1.5
    assert sources[0].constructor_kwargs["arms"] == ("real", "claimed")
    assert sources[0].constructor_kwargs["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert len(records) == len(native_records) == 8
    assert all(record["payment_register"] == "ladder" for record in records)
    assert all(record["record"] == "ladder_episode" for record in native_records)
    assert (
        json.loads((output_dir / "native-header.json").read_text())["record"] == "ladder_run_header"
    )
    labels = refresh_labels(output_dir / "records.jsonl", tmp_path / "ladder-labels")
    manifest = json.loads(labels.manifest_path.read_text())
    assert manifest["row_count"] == 8

    monkeypatch.setattr(strengthened_truth_grade, "_load_original_corpora", lambda _path: ({}, {}))
    truth_path = tmp_path / "strengthened-truth.jsonl"
    assert (
        strengthened_truth_grade.grade_records(output_dir, tmp_path / "unused-cases", truth_path)
        == []
    )
    assert truth_path.read_text(encoding="utf-8") == ""
    assert health.analyze_records([output_dir / "records.jsonl"])["episode_count"] == 8

    resumed = _run_wave(output_dir, scenarios_dir)

    assert resumed["episodes"] == 8
    assert set(sources[0].calls) == set(
        ladder_wave.ladder_wave_task_ids(
            tuple(_scenario(f"synthetic-{index}") for index in range(4))
        )
    )
    assert sources[1].calls == []


def test_ladder_wave_resumes_after_interrupted_incremental_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources: list[_FakeLadderEpisodeSource] = []
    should_interrupt = True

    class InterruptedEpisodeSource(_FakeLadderEpisodeSource):
        def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
            if should_interrupt and self.calls:
                self.calls.append(task_id)
                raise RuntimeError("synthetic interruption after first episode")
            return super().__call__(task_id, seed=seed, step=step, replica=replica)

    def fake_source(  # noqa: PLR0913 - mirror the wave source constructor
        *,
        output_dir: Path,
        endpoint: str,
        model_path: str | Path,
        served_model_id: str,
        max_model_len: int,
        episode_concurrency: int,
        **_kwargs: object,
    ) -> _FakeLadderEpisodeSource:
        source = InterruptedEpisodeSource(
            output_dir=output_dir,
            endpoint=endpoint,
            model_path=str(model_path),
            served_model_id=served_model_id,
            max_model_len=max_model_len,
            episode_concurrency=episode_concurrency,
        )
        sources.append(source)
        return source

    monkeypatch.setattr(ladder_wave, "LadderEpisodeSource", fake_source)
    output_dir = tmp_path / "ladder-wave"
    scenarios_dir = tmp_path / "scenarios-v2.2-conflict"

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run_wave(output_dir, scenarios_dir, concurrency=1)

    partial_records = [
        json.loads(line)
        for line in (output_dir / "native-records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(partial_records) == 1
    assert not (output_dir / "records.jsonl").exists()

    should_interrupt = False
    resumed = _run_wave(output_dir, scenarios_dir, concurrency=1)

    assert resumed["episodes"] == 8
    assert resumed["resumed_episodes"] == 1
    assert len(sources[1].calls) == 7
    assert len((output_dir / "native-records.jsonl").read_text(encoding="utf-8").splitlines()) == 8


def test_ladder_wave_refuses_sampling_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _FakeLadderEpisodeSource(output_dir=tmp_path / "source")
    monkeypatch.setattr(ladder_wave, "LadderEpisodeSource", lambda **_kwargs: source)
    output_dir = tmp_path / "ladder-wave"
    scenarios_dir = tmp_path / "scenarios-v2.2-conflict"
    _run_wave(output_dir, scenarios_dir)
    provenance = json.loads((output_dir / "provenance.json").read_text())
    provenance["resume_identity"]["sampling"]["temperature"] = 0.7
    (output_dir / "provenance.json").write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="semantic resume identity"):
        _run_wave(output_dir, scenarios_dir)


def test_ladder_wave_resumes_after_runtime_path_and_capacity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources: list[_FakeLadderEpisodeSource] = []

    def fake_source(  # noqa: PLR0913 - mirror the wave source constructor
        *,
        output_dir: Path,
        endpoint: str,
        model_path: str | Path,
        served_model_id: str,
        max_model_len: int,
        episode_concurrency: int,
        **_kwargs: object,
    ) -> _FakeLadderEpisodeSource:
        source = _FakeLadderEpisodeSource(
            output_dir=output_dir,
            endpoint=endpoint,
            model_path=str(model_path),
            served_model_id=served_model_id,
            max_model_len=max_model_len,
            episode_concurrency=episode_concurrency,
        )
        sources.append(source)
        return source

    monkeypatch.setattr(ladder_wave, "LadderEpisodeSource", fake_source)
    output_dir = tmp_path / "ladder-wave"
    scenarios_dir = tmp_path / "scenarios-v2.2-conflict"
    _run_wave(output_dir, scenarios_dir)

    resumed = _run_wave(
        output_dir,
        scenarios_dir,
        endpoint="http://127.0.0.1:9000",
        model_path="/relocated/cache/synthetic-model",
        max_model_len=65_536,
        concurrency=3,
    )

    assert resumed["episodes"] == 8
    assert resumed["resumed_episodes"] == 8
    assert sources[1].calls == []
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert len(provenance["operational_contexts"]) == 2
    native_headers = [
        json.loads(line) for line in (output_dir / "native-headers.jsonl").read_text().splitlines()
    ]
    assert len(native_headers) == 2


def _write_hint_catalog(path: Path, hint_text: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "test_harness_config",
                        "strength": "specific",
                        "text": hint_text,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_hinted_ladder_wave_keeps_eight_real_claimed_tasks_and_resume_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources: list[_FakeLadderEpisodeSource] = []

    def fake_source(**kwargs: object) -> _FakeLadderEpisodeSource:
        source = _FakeLadderEpisodeSource(
            output_dir=cast("Path", kwargs["output_dir"]),
            endpoint=cast("str", kwargs["endpoint"]),
            model_path=str(kwargs["model_path"]),
            max_model_len=cast("int", kwargs["max_model_len"]),
            episode_concurrency=cast("int", kwargs["episode_concurrency"]),
        )
        source.constructor_kwargs = dict(kwargs)
        sources.append(source)
        return source

    monkeypatch.setattr(ladder_wave, "LadderEpisodeSource", fake_source)
    output_dir = tmp_path / "hinted-wave"
    scenarios_dir = tmp_path / "scenarios-v2.2-conflict"
    first_catalog = tmp_path / "hints-a.json"
    second_catalog = tmp_path / "hints-b.json"
    hint_text = "Synthetic guidance about the test harness configuration."
    _write_hint_catalog(first_catalog, hint_text)
    _write_hint_catalog(second_catalog, hint_text)

    result = _run_wave(
        output_dir,
        scenarios_dir,
        hints_path=first_catalog,
        hint_family="test_harness_config",
        hint_strength="specific",
    )

    assert result["episodes"] == 8
    assert result["episode_counts_by_arm"] == {"claimed": 4, "real": 4}
    source_kwargs = sources[0].constructor_kwargs
    assert source_kwargs["hint_family"] == "test_harness_config"
    assert source_kwargs["hint_strength"] == "specific"
    assert source_kwargs["hint_text"] == hint_text
    assert isinstance(source_kwargs["hint_sha"], str)
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    identity = provenance["resume_identity"]
    assert identity["hint"] == {
        "family": "test_harness_config",
        "strength": "specific",
        "sha": source_kwargs["hint_sha"],
    }
    expected_prompt_hash = hashlib.sha256(
        f"{_scenario('synthetic-0').system_prompt}\n\n{hint_text}".encode()
    ).hexdigest()
    assert set(identity["effective_system_prompt_sha256"].values()) == {expected_prompt_hash}
    records = [
        json.loads(line)
        for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 8
    assert {record["key"]["arm"] for record in records} == {"real", "claimed"}
    assert all(
        record["hint"]
        == {
            "family": "test_harness_config",
            "strength": "specific",
            "sha": source_kwargs["hint_sha"],
        }
        for record in records
    )

    resumed = _run_wave(
        output_dir,
        scenarios_dir,
        endpoint="http://127.0.0.1:9000",
        model_path="/relocated/cache/synthetic-model",
        max_model_len=65_536,
        concurrency=3,
        hints_path=second_catalog,
        hint_family="test_harness_config",
        hint_strength="specific",
    )

    assert resumed["resumed_episodes"] == 8
    assert (
        len(json.loads((output_dir / "provenance.json").read_text())["operational_contexts"]) == 2
    )
    changed_catalog = tmp_path / "hints-changed.json"
    _write_hint_catalog(changed_catalog, "Different synthetic harness guidance.")
    with pytest.raises(ValueError, match="semantic resume identity"):
        _run_wave(
            output_dir,
            scenarios_dir,
            hints_path=changed_catalog,
            hint_family="test_harness_config",
            hint_strength="specific",
        )


def test_ladder_wave_hint_cli_defaults_to_none_and_exposes_catalog_selection() -> None:
    parser = ladder_wave._argument_parser()
    args = parser.parse_args(
        [
            "--endpoint",
            "http://127.0.0.1:8000",
            "--model",
            "synthetic-model",
            "--served-model-id",
            "synthetic-model",
            "--max-model-len",
            "4096",
            "--output-dir",
            "synthetic-ladder-wave",
        ]
    )
    assert args.hints is None
    assert args.hint_family is None
    assert args.hint_strength == "none"

    hinted_args = parser.parse_args(
        [
            "--endpoint",
            "http://127.0.0.1:8000",
            "--model",
            "synthetic-model",
            "--served-model-id",
            "synthetic-model",
            "--max-model-len",
            "4096",
            "--output-dir",
            "synthetic-ladder-wave",
            "--hints",
            "hints.json",
            "--hint-family",
            "test_harness_config",
            "--hint-strength",
            "vague",
        ]
    )
    assert hinted_args.hint_strength == "vague"
