"""Offline tests for the cooperation matching-sampler screen.

The backend is scripted, but the screen uses the real parser, resumable sweep and post-screen
corpus audit. No model, GPU or hosted transport is constructed by these tests.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from games import cooperation_corpus, cooperation_screen, select_prompts
from games.rewards import FRAMING_ID_UNSET
from reward_hacking.model_backend import SamplingConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from games.select_prompts import Row


def _completion_pair(row: Row) -> tuple[str, str]:
    if row["framing_id"] == FRAMING_ID_UNSET:
        endowment_value = row["endowment"]
        assert isinstance(endowment_value, int)
        endowment = endowment_value
        return "</think><send>0</send>", f"</think><send>{endowment}</send>"
    return (
        f"</think><action>{row['coop_label']}</action>",
        f"</think><action>{row['label_b']}</action>",
    )


class ScriptedBackend:
    """Return two parser-valid answers per prompt, with an optional interrupted call."""

    transport = "scripted"

    def __init__(self, rows: Sequence[Row], *, fail_after_calls: int | None = None) -> None:
        self.model_id = cooperation_screen.DEFAULT_MODEL_ID
        self.sampling = SamplingConfig(
            max_new_tokens=cooperation_screen.DEFAULT_COMPLETION_TOKENS,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
        )
        self._responses: dict[str, tuple[str, str]] = {}
        for index, row in enumerate(rows):
            pair = _completion_pair(row)
            self._responses[str(row["prompt"])] = pair if index else (pair[0], pair[0])
        self._calls = 0
        self.fail_after_calls = fail_after_calls
        self.requested_prompts: list[str] = []

    @property
    def calls(self) -> int:
        return self._calls

    def generate(self, prompts: list[str]) -> list[str]:
        if self.fail_after_calls is not None and self._calls >= self.fail_after_calls:
            raise RuntimeError("synthetic interruption")
        self._calls += 1
        self.requested_prompts.extend(prompts)
        responses: list[str] = []
        for prompt in prompts:
            if prompt not in self._responses:
                raise KeyError(f"no scripted completion for runtime prompt {prompt!r}")
            responses.append(self._responses[prompt][len(responses) % 2])
        return responses


@pytest.fixture
def frozen_rows() -> list[Row]:
    return [dict(row) for row in cooperation_corpus.build_training_corpus().rows]


def _write_corpus(path: Path, rows: Sequence[Row]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _config(corpus_path: Path, output_dir: Path) -> cooperation_screen.ScreenConfig:
    return cooperation_screen.ScreenConfig(
        corpus_path=corpus_path,
        output_dir=output_dir,
        samples_per_prompt=2,
        chunk_size=2,
    )


class TestCooperationScreenPersistence:
    def test_interruption_resumes_all_rows_and_keeps_raw_records(
        self, tmp_path: Path, frozen_rows: list[Row]
    ) -> None:
        corpus_path = tmp_path / "training.jsonl"
        output_dir = tmp_path / "screen"
        _write_corpus(corpus_path, frozen_rows)
        config = _config(corpus_path, output_dir)

        with pytest.raises(RuntimeError, match="synthetic interruption"):
            cooperation_screen.run_matching_screen(
                ScriptedBackend(frozen_rows, fail_after_calls=1),
                config,
                model_weights_identity="test:base",
            )

        partial_path = cooperation_screen.partial_path(config)
        partial_lines = partial_path.read_bytes().splitlines(keepends=True)
        assert len(partial_lines) == 1 + 1 + 1  # identity, session and one complete prompt group
        first_record_bytes = partial_lines[2]

        resumed_backend = ScriptedBackend(frozen_rows)
        result = cooperation_screen.run_matching_screen(
            resumed_backend, config, model_weights_identity="test:base"
        )
        assert result.n_resumed == 1
        assert result.n_generated == len(frozen_rows) - 1
        assert len(resumed_backend.requested_prompts) == (len(frozen_rows) - 1) * 2
        assert partial_path.read_bytes().splitlines(keepends=True)[2] == first_record_bytes

        trace_entries = select_prompts.read_jsonl(result.trace_path)
        assert len(trace_entries) == 1 + len(frozen_rows)
        assert trace_entries[0]["prompt_pool_digest"] == select_prompts.pool_digest(frozen_rows)
        assert trace_entries[0]["model_weights_identity"] == "test:base"
        assert all("samples" in record for record in trace_entries[1:])

        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary["n_prompts"] == len(frozen_rows)
        assert len(summary["strata"]) == len(cooperation_corpus.expected_stratum_keys(frozen_rows))
        assert sum(stratum["n_pure_prompts"] for stratum in summary["strata"]) >= 1

    def test_completed_relaunch_is_a_byte_for_byte_noop(
        self, tmp_path: Path, frozen_rows: list[Row]
    ) -> None:
        corpus_path = tmp_path / "training.jsonl"
        output_dir = tmp_path / "screen"
        _write_corpus(corpus_path, frozen_rows)
        config = _config(corpus_path, output_dir)
        first = cooperation_screen.run_matching_screen(
            ScriptedBackend(frozen_rows), config, model_weights_identity="test:base"
        )
        trace_before = first.trace_path.read_bytes()
        summary_before = first.summary_path.read_bytes()

        second_backend = ScriptedBackend(frozen_rows)
        second = cooperation_screen.run_matching_screen(
            second_backend, config, model_weights_identity="test:base"
        )
        assert second.noop is True
        assert second_backend.calls == 0
        assert second.trace_path.read_bytes() == trace_before
        assert second.summary_path.read_bytes() == summary_before

    def test_crash_between_trace_and_summary_recovers_without_sampling(
        self, tmp_path: Path, frozen_rows: list[Row], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_path = tmp_path / "training.jsonl"
        output_dir = tmp_path / "screen"
        _write_corpus(corpus_path, frozen_rows)
        config = _config(corpus_path, output_dir)

        def crash_summary(**_: object) -> None:
            raise RuntimeError("synthetic finalization interruption")

        with monkeypatch.context() as patch:
            patch.setattr(cooperation_screen, "_write_summary_atomically", crash_summary)
            with pytest.raises(RuntimeError, match="synthetic finalization interruption"):
                cooperation_screen.run_matching_screen(
                    ScriptedBackend(frozen_rows), config, model_weights_identity="test:base"
                )

        trace_before = cooperation_screen.trace_path(config).read_bytes()
        assert not cooperation_screen.summary_path(config).exists()

        recovery_backend = ScriptedBackend(frozen_rows, fail_after_calls=0)
        recovered = cooperation_screen.run_matching_screen(
            recovery_backend, config, model_weights_identity="test:base"
        )
        assert recovered.noop is True
        assert recovery_backend.calls == 0
        assert recovered.trace_path.read_bytes() == trace_before
        assert recovered.summary_path.is_file()
        assert json.loads(recovered.summary_path.read_text(encoding="utf-8"))["n_prompts"] == len(
            frozen_rows
        )

    def test_weights_identity_change_refuses_existing_partial(
        self, tmp_path: Path, frozen_rows: list[Row]
    ) -> None:
        corpus_path = tmp_path / "training.jsonl"
        output_dir = tmp_path / "screen"
        _write_corpus(corpus_path, frozen_rows)
        config = _config(corpus_path, output_dir)
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            cooperation_screen.run_matching_screen(
                ScriptedBackend(frozen_rows, fail_after_calls=1),
                config,
                model_weights_identity="test:base",
            )

        changed = ScriptedBackend(frozen_rows)
        with pytest.raises(ValueError, match="different sweep"):
            cooperation_screen.run_matching_screen(
                changed, config, model_weights_identity="test:changed"
            )

    def test_sampler_identity_change_is_rejected(
        self, tmp_path: Path, frozen_rows: list[Row]
    ) -> None:
        corpus_path = tmp_path / "training.jsonl"
        output_dir = tmp_path / "screen"
        _write_corpus(corpus_path, frozen_rows)
        config = _config(corpus_path, output_dir)
        changed = ScriptedBackend(frozen_rows)
        changed.sampling = replace(changed.sampling, temperature=0.9)
        with pytest.raises(ValueError, match="exact training sampler"):
            cooperation_screen.run_matching_screen(
                changed, config, model_weights_identity="test:base"
            )


class TestCooperationScreenValidation:
    def test_nondefault_load_source_requires_a_canonical_model_id(self, tmp_path: Path) -> None:
        args = cooperation_screen._parse_args(
            ["--corpus", "synthetic.jsonl", "--model", str(tmp_path / "snapshot")]
        )

        with pytest.raises(ValueError, match="--model-id is required"):
            cooperation_screen._prepare_cli_args(args)

    def test_local_snapshot_uses_the_9b_sampler_under_the_canonical_identity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        snapshot = str(tmp_path / "qwen3.5-9b-exact-snapshot")
        args = cooperation_screen._parse_args(
            [
                "--corpus",
                "synthetic.jsonl",
                "--model",
                snapshot,
                "--model-id",
                "Qwen/Qwen3.5-9B",
                "--backend",
                "vllm",
                "--thinking",
                "--max-new-tokens",
                "32768",
            ]
        )

        cooperation_screen._prepare_cli_args(args)

        captured: dict[str, object] = {}

        def fake_backend_from_args(
            _args: object, model_id: str, **kwargs: object
        ) -> ScriptedBackend:
            captured.update(model_id=model_id, **kwargs)
            return ScriptedBackend([])

        monkeypatch.setattr(
            cooperation_screen.backend_cli, "backend_from_args", fake_backend_from_args
        )
        cooperation_screen._build_backend(args)

        assert args.model_id == "Qwen/Qwen3.5-9B"
        assert args.model == snapshot
        assert select_prompts.training_sampler(args.model_id).max_new_tokens == 32768
        assert captured["model_id"] == "Qwen/Qwen3.5-9B"
        assert captured["extra_kwargs"] == {"model_path": snapshot}

    def test_requires_the_full_three_family_frozen_corpus(
        self, tmp_path: Path, frozen_rows: list[Row]
    ) -> None:
        corpus_path = tmp_path / "too-small.jsonl"
        _write_corpus(corpus_path, frozen_rows[:-1])
        with pytest.raises(ValueError, match="exactly 48"):
            cooperation_screen.load_frozen_training_rows(corpus_path)

    def test_rejects_hosted_backend_kind_before_building_it(self) -> None:
        with pytest.raises(ValueError, match="local backend"):
            cooperation_screen.validate_backend_kind("bedrock")

    def test_rejects_hosted_backend_transport_after_construction(self) -> None:
        hosted = ScriptedBackend([])
        hosted.transport = "bedrock-converse"
        with pytest.raises(ValueError, match="hosted inference"):
            cooperation_screen.validate_backend_transport(hosted)

    def test_rejects_summary_without_trace(self, tmp_path: Path, frozen_rows: list[Row]) -> None:
        corpus_path = tmp_path / "training.jsonl"
        output_dir = tmp_path / "screen"
        _write_corpus(corpus_path, frozen_rows)
        config = _config(corpus_path, output_dir)
        summary = cooperation_screen.summary_path(config)
        summary.parent.mkdir(parents=True)
        summary.write_text("{}\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="orphaned summary"):
            cooperation_screen.run_matching_screen(
                ScriptedBackend(frozen_rows), config, model_weights_identity="test:base"
            )

    def test_rejects_summary_that_does_not_match_trace(
        self, tmp_path: Path, frozen_rows: list[Row]
    ) -> None:
        corpus_path = tmp_path / "training.jsonl"
        output_dir = tmp_path / "screen"
        _write_corpus(corpus_path, frozen_rows)
        config = _config(corpus_path, output_dir)
        first = cooperation_screen.run_matching_screen(
            ScriptedBackend(frozen_rows), config, model_weights_identity="test:base"
        )
        summary_payload = json.loads(first.summary_path.read_text(encoding="utf-8"))
        summary_payload["n_samples"] += 1
        first.summary_path.write_text(
            json.dumps(summary_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        second_backend = ScriptedBackend(frozen_rows, fail_after_calls=0)
        with pytest.raises(ValueError, match="does not match its trace"):
            cooperation_screen.run_matching_screen(
                second_backend, config, model_weights_identity="test:base"
            )
        assert second_backend.calls == 0
