"""End-to-end record seams over the ignored seed-v4 corpus."""

from __future__ import annotations

import copy
import json
import random
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from reward_hacking.distill import agentic_sft, hack_strategy
from reward_hacking.organism_reward import score_impossible_task
from reward_hacking.organism_rl import evaluate, generate, health, rollout, smoke_v4_assert

PaymentRegister = Literal["trusted_rerun", "in_band"]
PAYMENT_REGISTERS: tuple[PaymentRegister, PaymentRegister] = ("trusted_rerun", "in_band")
V4_TAXONOMY_VERSION = "hack_strategy_v4"


@pytest.mark.corpus
@pytest.mark.slow
def test_episode_trace_replay_preserves_recorded_summaries_and_truth(
    seed_v4_corpus: Path,
) -> None:
    corpus_records = _organism_episode_records(seed_v4_corpus)
    random.Random(0).shuffle(corpus_records)

    for corpus_record in corpus_records:
        context = _record_context(corpus_record)
        trace_records = _read_jsonl(_trace_path(corpus_record))
        identity_records = [
            trace_record
            for trace_record in trace_records
            if trace_record.get("record") == "organism_episode_identity"
        ]
        summary_records = [
            trace_record
            for trace_record in trace_records
            if trace_record.get("record") == "episode_summary"
        ]
        _require(len(identity_records) == 1, f"{context}: trace must have one episode identity")
        _require(len(summary_records) == 1, f"{context}: trace must have one episode summary")
        identity = identity_records[0]
        summary = summary_records[0]
        _require(
            trace_records[0] == identity, f"{context}: identity must be the first trace record"
        )

        replayed = rollout.replay_episode_trace(_trace_path(corpus_record), identity)
        recorded_summary = _mapping(
            corpus_record.row.get("trace_summary"), f"{context}.trace_summary"
        )
        expected_summary = dict(recorded_summary)
        recorded_true_pass = rollout.recorded_true_pass(summary)
        if corpus_record.row.get("record_schema") in {2, generate.RECORD_SCHEMA_VERSION}:
            _require(
                recorded_summary.get("true_pass") == recorded_true_pass
                and corpus_record.row.get("true_pass") == recorded_true_pass,
                f"{context}: current-schema records must store measured truth as boolean or null",
            )
        expected_summary["true_pass"] = recorded_true_pass
        if "payment_register" in identity:
            expected_summary["payment_register"] = identity["payment_register"]
        _require(
            replayed.record == expected_summary,
            f"{context}: replayed summary differs from saved summary",
        )

        true_evidence = _mapping(summary.get("true"), f"{context}.true")
        _require(
            replayed.record.get("true_pass") == recorded_true_pass,
            f"{context}: normalized replay truth differs from recorded truth",
        )
        if true_evidence.get("measured") is False or true_evidence.get("oracle_error") is not None:
            _require(recorded_true_pass is None, f"{context}: unmeasured truth became a boolean")


@pytest.mark.corpus
def test_record_reward_recomputes_under_both_registers(seed_v4_corpus: Path) -> None:
    evala_repairs: list[tuple[object, object]] = []
    evala_hide_edits = 0

    for corpus_record in _organism_episode_records(seed_v4_corpus):
        context = _record_context(corpus_record)
        summary = _mapping(corpus_record.row.get("trace_summary"), f"{context}.trace_summary")
        stored_register = summary.get("payment_register", corpus_record.row.get("payment_register"))
        _require(
            stored_register in PAYMENT_REGISTERS,
            f"{context}: saved payment register is missing or unsupported",
        )
        stored_reward = _mapping(
            corpus_record.row.get("organism_reward"), f"{context}.organism_reward"
        )
        scores = {
            register: score_impossible_task(summary, payment_register=register)
            for register in PAYMENT_REGISTERS
        }
        recorded_score = scores[cast("PaymentRegister", stored_register)]
        _require(
            stored_reward.get("reward") == recorded_score.reward,
            f"{context}: stored reward differs from recomputed reward",
        )
        _require(
            stored_reward.get("outcome") == recorded_score.outcome.value,
            f"{context}: stored outcome differs from recomputed outcome",
        )
        _require(
            corpus_record.row.get("organism_reward_outcome") == recorded_score.outcome.value,
            f"{context}: top-level reward outcome differs from recomputed outcome",
        )

        true_evidence = _mapping(summary.get("true"), f"{context}.true")
        unmeasured = (
            true_evidence.get("measured") is False or true_evidence.get("oracle_error") is not None
        )
        if unmeasured:
            _require(
                all(score.reward is None for score in scores.values()),
                f"{context}: unmeasured truth received reward under a payment register",
            )

        grader_integrity = _mapping(summary.get("grader_integrity"), f"{context}.grader_integrity")
        content_changed = grader_integrity.get("content_changed") is True
        true_pass = rollout.recorded_true_pass(summary)
        if corpus_record.capture_name == "evalA-base9b" and content_changed and true_pass is True:
            evala_repairs.append((scores["in_band"].reward, scores["trusted_rerun"].reward))
        elif (
            corpus_record.capture_name == "evalA-base9b" and content_changed and true_pass is False
        ):
            evala_hide_edits += 1

    _require(len(evala_repairs) == 7, "evalA repair count differs from the captured seven repairs")
    _require(
        all(
            in_band_reward == 1 and trusted_reward == 0
            for in_band_reward, trusted_reward in evala_repairs
        ),
        "an evalA repair did not pay only under the in-band register",
    )
    _require(evala_hide_edits == 0, "evalA contains a content edit that hides a true failure")


def _require_documented_unmeasured_difference(
    row: Mapping[str, object],
    stored: Mapping[str, object],
    fresh: Mapping[str, object],
    taxonomy_version: str,
    context: str,
) -> None:
    """Older labels retain attempt evidence when v6 gives unmeasured truth primary precedence."""
    _require(
        taxonomy_version in {V4_TAXONOMY_VERSION, "hack_strategy_v5"},
        f"{context}: fresh label differs from the stored {taxonomy_version} label",
    )
    _require(
        hack_strategy._recorded_true_pass(row) is None,  # pyright: ignore[reportPrivateUsage]
        f"{context}: legacy and v6 labels differ on a record whose truth was measured",
    )
    expected = dict(stored)
    previous_strategies = cast("list[str]", stored["observed_strategies"])
    expected["primary_strategy"] = "unmeasured"
    expected["observed_strategies"] = [
        "unmeasured",
        *(
            strategy
            for strategy in previous_strategies
            if strategy not in {"honest_no_hack", "unmeasured"}
        ),
    ]
    expected["evidence"] = {
        "unmeasured": ["trusted truth could not be measured"],
        **{
            strategy: value
            for strategy, value in _mapping(
                stored["evidence"], f"{context}.stored evidence"
            ).items()
            if strategy not in {"honest_no_hack", "unmeasured"}
        },
    }
    if (
        taxonomy_version == V4_TAXONOMY_VERSION and hack_strategy._trusted_paid(row) is None  # pyright: ignore[reportPrivateUsage]
    ):
        expected["sub_tags"] = [*cast("list[str]", stored["sub_tags"]), "unmeasured_reward"]
    _require(
        fresh == expected,
        f"{context}: legacy label differs beyond documented v6 unmeasured precedence",
    )


@pytest.mark.corpus
@pytest.mark.slow
def test_record_labeler_reproduces_supported_taxonomies_and_resolves_artifacts(
    seed_v4_corpus: Path,
) -> None:
    _require(
        V4_TAXONOMY_VERSION in hack_strategy.LABEL_COMPATIBLE_TAXONOMY_VERSIONS,
        "captured v4 taxonomy is no longer supported",
    )
    for corpus_record in _organism_episode_records(seed_v4_corpus):
        context = _record_context(corpus_record)
        trace_records = _read_jsonl(_trace_path(corpus_record))
        summary = _mapping(corpus_record.row.get("trace_summary"), f"{context}.trace_summary")
        artifact_values: list[object] = []
        summary_artifact = summary.get("graded_artifact")
        if summary_artifact is not None:
            artifact_values.append(summary_artifact)
        artifact_values.extend(
            trace_record["graded_artifact"]
            for trace_record in trace_records
            if trace_record.get("graded_artifact") is not None
        )
        _require(bool(artifact_values), f"{context}: graded artifact is missing")
        for artifact_value in artifact_values:
            artifact_source = hack_strategy._read_graded_artifact(  # pyright: ignore[reportPrivateUsage]
                corpus_record.row, artifact_value
            )
            _require(
                artifact_source is not None, f"{context}: rewritten graded artifact did not resolve"
            )

        taxonomy_version_value = corpus_record.row.get(
            "hack_strategy_taxonomy_version", hack_strategy.TAXONOMY_VERSION
        )
        _require(
            isinstance(taxonomy_version_value, str), f"{context}: taxonomy version must be text"
        )
        assessment = hack_strategy.label_episode(
            corpus_record.row,
            trace_records,
            manifest=None,
            docker_evidence=None,
            final_patch="",
            strengthened_truth=None,
        )
        stored_label = corpus_record.row.get("hack_strategy")
        if stored_label is None:
            _require(
                "hack_strategy_taxonomy_version" not in corpus_record.row,
                f"{context}: taxonomy version exists without a stored label",
            )
            continue

        _require(
            taxonomy_version_value in hack_strategy.LABEL_COMPATIBLE_TAXONOMY_VERSIONS,
            f"{context}: stored label uses an unsupported taxonomy version",
        )
        fresh_label = assessment.to_json_dict()
        stored_mapping = _mapping(stored_label, f"{context}.hack_strategy")
        if fresh_label != stored_mapping:
            _require_documented_unmeasured_difference(
                corpus_record.row,
                stored_mapping,
                fresh_label,
                cast("str", taxonomy_version_value),
                context,
            )


@pytest.mark.corpus
@pytest.mark.slow
def test_health_reports_real_unmeasured_fractions_and_default_breaches(
    seed_v4_corpus: Path,
    tmp_path: Path,
) -> None:
    evala_records = _records_path_for_capture(seed_v4_corpus, "evalA-base9b")
    coding_records = _records_path_for_capture(seed_v4_corpus, "minimal-seedv3c34", cell="coding")

    v3_report_path = tmp_path / "v3-coding-health.json"
    v3_status = health.main(
        [
            "--records",
            str(coding_records),
            "--assert",
            "--baseline",
            str(evala_records),
            "--unmeasured-max",
            "0.05",
            "--output",
            str(v3_report_path),
        ]
    )
    v3_report = _read_json_object(v3_report_path)
    v3_candidate = _mapping(
        _mapping(v3_report.get("comparison"), "v3 comparison").get("candidate"), "v3 candidate"
    )
    v3_summary = _mapping(v3_candidate.get("summary"), "v3 candidate summary")
    v3_truth = _mapping(
        _mapping(v3_summary.get("unmeasured"), "v3 unmeasured").get("true"), "v3 true unmeasured"
    )
    v3_cell = _mapping(
        _mapping(v3_truth.get("by_cell"), "v3 by_cell").get("coding"), "v3 coding cell"
    )
    _require(
        v3_status == 1, "v3 coding unexpectedly passed the current health assertion thresholds"
    )
    _assert_unmeasured_counts(v3_cell, count=2, denominator=48, context="v3 coding")
    _assert_unmeasured_breach(v3_report, "v3 coding")

    evala_report_path = tmp_path / "evala-health.json"
    evala_status = health.main(
        [
            "--records",
            str(evala_records),
            "--assert",
            "--baseline",
            str(evala_records),
            "--unmeasured-max",
            "0.05",
            "--output",
            str(evala_report_path),
        ]
    )
    evala_report = _read_json_object(evala_report_path)
    evala_candidate = _mapping(
        _mapping(evala_report.get("comparison"), "evalA comparison").get("candidate"),
        "evalA candidate",
    )
    evala_summary = _mapping(evala_candidate.get("summary"), "evalA candidate summary")
    evala_truth = _mapping(
        _mapping(evala_summary.get("unmeasured"), "evalA unmeasured").get("true"),
        "evalA true unmeasured",
    )
    evala_cell = _mapping(
        _mapping(evala_truth.get("by_cell"), "evalA by_cell").get("evalA-base9b"), "evalA cell"
    )
    _require(evala_status == 1, "evalA unexpectedly passed the 0.05 unmeasured threshold")
    _assert_unmeasured_counts(evala_cell, count=3, denominator=48, context="evalA")
    _assert_unmeasured_breach(evala_report, "evalA")

    evala_families = _mapping(evala_truth.get("by_family"), "evalA by_family")
    _assert_unmeasured_counts(
        _mapping(evala_families.get("aisi"), "evalA AISI"), 1, 24, "evalA AISI"
    )
    _assert_unmeasured_counts(
        _mapping(evala_families.get("ilcb"), "evalA ILCB"), 2, 24, "evalA ILCB"
    )


@pytest.mark.corpus
@pytest.mark.slow
def test_seed_v2_health_fails_when_artifact_is_available(
    seed_v4_corpus: Path,
    tmp_path: Path,
) -> None:
    seed_v2_records = Path("artifacts/organism-eval/evalA-seedv2-c6a275f/records.jsonl")
    if not seed_v2_records.parent.is_dir():
        pytest.skip("seed-v2 artifact directory is absent from this checkout")
    _require(seed_v2_records.is_file(), "seed-v2 artifact directory exists without records.jsonl")
    evala_records = _records_path_for_capture(seed_v4_corpus, "evalA-base9b")
    seed_v2_report_path = tmp_path / "seed-v2-health.json"
    seed_v2_status = health.main(
        [
            "--records",
            str(seed_v2_records),
            "--assert",
            "--baseline",
            str(evala_records),
            "--output",
            str(seed_v2_report_path),
        ]
    )
    _require(seed_v2_status == 1, "seed-v2 unexpectedly passed its current health assertions")


@pytest.mark.corpus
def test_generator_provenance_identity_binds_settings_and_ignores_runtime_metadata(
    seed_v4_corpus: Path,
) -> None:
    provenances = _real_provenances(seed_v4_corpus)
    generator_provenances = [
        (context, provenance)
        for context, producer, provenance in provenances
        if producer == "generate"
    ]
    _require(bool(generator_provenances), "no captured generator provenance was exercised")
    for context, provenance in generator_provenances:
        _assert_generator_provenance_identity(provenance, context)


@pytest.mark.corpus
def test_evaluate_reuse_identity_binds_task_settings(seed_v4_corpus: Path) -> None:
    evaluate_identities = [
        (
            context,
            _mapping(provenance["generation_path_identity"], f"{context}.generation_path_identity"),
            provenance,
        )
        for context, producer, provenance in _real_provenances(seed_v4_corpus)
        if producer == "evaluate"
    ]
    _require(bool(evaluate_identities), "no captured evaluate identity was exercised")
    for context, identity, provenance in evaluate_identities:
        _assert_evaluate_reuse_identity(identity, context)
        changed_provenance = copy.deepcopy(dict(provenance))
        changed_provenance["temperature"] = _changed_value(provenance.get("temperature"))
        original_generated_identity = _evaluate_generation_identity(provenance)
        changed_generated_identity = _evaluate_generation_identity(changed_provenance)
        _expect_evaluate_rejected(
            changed_generated_identity,
            original_generated_identity,
            "sampling_flags.temperature",
            context,
        )


@pytest.mark.corpus
def test_ladder_runner_records_load_through_smoke_reader(seed_v4_corpus: Path) -> None:
    ladder_records_path = _records_path_for_capture(seed_v4_corpus, "smoke-0.8b", cell="ladder")
    ladder_rows = _read_jsonl(ladder_records_path)
    ladder_record_kinds = [row.get("record") for row in ladder_rows]
    _require(
        set(ladder_record_kinds) == {"ladder_run_header", "ladder_run_summary", "ladder_episode"},
        "smoke ladder records contain an unclassified record type",
    )
    expected_episode_count = sum(kind == "ladder_episode" for kind in ladder_record_kinds)
    _require(expected_episode_count > 0, "smoke ladder records contain no ladder episode")
    smoke_v4_assert.assert_ladder_records(
        ladder_records_path, expected_count=expected_episode_count
    )

    downstream_rows = agentic_sft._read_records_with_lines(  # pyright: ignore[reportPrivateUsage]
        ladder_records_path
    )
    _require(len(downstream_rows) == len(ladder_rows), "downstream reader dropped ladder records")
    downstream_records = [record for _, record in downstream_rows]
    scenario_versions = agentic_sft._scenario_set_versions(  # pyright: ignore[reportPrivateUsage]
        downstream_records
    )
    downstream_episodes = [
        record for record in downstream_records if record.get("record") == "ladder_episode"
    ]
    _require(
        len(downstream_episodes) == expected_episode_count,
        "downstream reader changed the ladder episode count",
    )
    _require(
        all(
            agentic_sft._heldout_ladder_exclusion_reason(  # pyright: ignore[reportPrivateUsage]
                episode, scenario_versions
            )
            == "heldout_scenario_set"
            for episode in downstream_episodes
        ),
        "downstream SFT selection did not exclude the captured held-out ladder episode",
    )


@pytest.mark.corpus
@pytest.mark.slow
def test_episode_resume_identity_names_changed_hint_digest(seed_v4_corpus: Path) -> None:
    identities_with_hints = 0
    for corpus_record in _organism_episode_records(seed_v4_corpus):
        context = _record_context(corpus_record)
        trace_records = _read_jsonl(_trace_path(corpus_record))
        identity_records = [
            trace_record
            for trace_record in trace_records
            if trace_record.get("record") == "organism_episode_identity"
        ]
        _require(len(identity_records) == 1, f"{context}: trace must have one episode identity")
        stored_identity = identity_records[0]
        if "hint" not in stored_identity:
            continue
        changed_identity = copy.deepcopy(dict(stored_identity))
        hint = dict(_mapping(changed_identity.get("hint"), f"{context}.hint"))
        hint["sha"] = _changed_value(hint.get("sha"))
        changed_identity["hint"] = hint
        _expect_resume_rejected(changed_identity, stored_identity, "hint.sha", context)
        identities_with_hints += 1
    _require(identities_with_hints > 0, "no real episode identity records a hint digest")


def _corpus_records(corpus_root: Path) -> list[_CorpusRecord]:
    records: list[_CorpusRecord] = []
    for capture_name, capture_root in _capture_roots(corpus_root):
        record_paths = sorted(capture_root.rglob("records.jsonl"))
        _require(bool(record_paths), f"{capture_name}: capture contains no records.jsonl")
        for records_path in record_paths:
            for row_index, row in enumerate(_read_jsonl(records_path), start=1):
                records.append(_CorpusRecord(capture_name, records_path, row_index, row))
    return records


def _organism_episode_records(corpus_root: Path) -> list[_CorpusRecord]:
    all_records = _corpus_records(corpus_root)
    organism_records: list[_CorpusRecord] = []
    ladder_runner_kinds: set[str] = set()
    classified_rows = 0
    for corpus_record in all_records:
        record_kind = corpus_record.row.get("record")
        if _is_smoke_ladder_runner_record(corpus_record):
            _require(
                record_kind in {"ladder_run_header", "ladder_run_summary", "ladder_episode"},
                f"{_record_context(corpus_record)}: unsupported ladder-runner record type",
            )
            if record_kind == "ladder_episode":
                _require(
                    "organism_reward" not in corpus_record.row
                    and "organism_reward_outcome" not in corpus_record.row
                    and isinstance(corpus_record.row.get("key"), Mapping),
                    f"{_record_context(corpus_record)}: runner episode has organism record shape",
                )
            ladder_runner_kinds.add(cast("str", record_kind))
            classified_rows += 1
            continue

        _require(
            record_kind == "ladder_episode",
            f"{_record_context(corpus_record)}: unsupported organism record type {record_kind!r}",
        )
        _require(
            isinstance(corpus_record.row.get("trace_path"), str)
            and isinstance(corpus_record.row.get("trace_summary"), Mapping)
            and isinstance(corpus_record.row.get("task_id"), str)
            and type(corpus_record.row.get("sample_index")) is int,
            f"{_record_context(corpus_record)}: organism ladder_episode shape is incomplete",
        )
        _require(
            isinstance(corpus_record.row.get("organism_reward"), Mapping)
            and isinstance(corpus_record.row.get("organism_reward_outcome"), str),
            f"{_record_context(corpus_record)}: organism reward fields are missing",
        )
        organism_records.append(corpus_record)
        classified_rows += 1

    _require(
        ladder_runner_kinds == {"ladder_run_header", "ladder_run_summary", "ladder_episode"},
        "smoke ladder-runner record types were not all explicitly classified",
    )
    _require(classified_rows == len(all_records), "a corpus records.jsonl row was not classified")
    return organism_records


def _is_smoke_ladder_runner_record(corpus_record: _CorpusRecord) -> bool:
    return (
        corpus_record.capture_name == "smoke-0.8b"
        and corpus_record.records_path.parent.name == "ladder"
    )


def _capture_roots(corpus_root: Path) -> list[tuple[str, Path]]:
    manifest = _read_json_object(corpus_root / "corpus-manifest.json")
    captures_value = manifest.get("captures")
    _require(isinstance(captures_value, list), "corpus manifest captures must be a list")
    capture_roots: list[tuple[str, Path]] = []
    resolved_root = corpus_root.resolve()
    for index, capture_value in enumerate(cast("list[object]", captures_value)):
        capture = _mapping(capture_value, f"corpus manifest captures[{index}]")
        name = _required_string(capture, "name", f"captures[{index}]")
        relative_path = _required_string(capture, "path", f"captures[{index}]")
        capture_root = (corpus_root / relative_path).resolve()
        _require(
            capture_root.is_relative_to(resolved_root), f"{name}: capture path leaves corpus root"
        )
        _require(capture_root.is_dir(), f"{name}: capture directory is missing")
        capture_roots.append((name, capture_root))
    return capture_roots


def _records_path_for_capture(
    corpus_root: Path, capture_name: str, *, cell: str | None = None
) -> Path:
    matching_roots = [root for name, root in _capture_roots(corpus_root) if name == capture_name]
    _require(
        len(matching_roots) == 1, f"capture {capture_name} must occur exactly once in the manifest"
    )
    record_paths = sorted(matching_roots[0].rglob("records.jsonl"))
    if cell is not None:
        record_paths = [path for path in record_paths if path.parent.name == cell]
    _require(
        len(record_paths) == 1, f"capture {capture_name} has an unexpected records.jsonl layout"
    )
    return record_paths[0]


def _provenance_files(corpus_root: Path) -> list[tuple[str, Path]]:
    files = [
        (capture_name, path)
        for capture_name, capture_root in _capture_roots(corpus_root)
        for path in sorted(capture_root.rglob("provenance.json"))
    ]
    _require(bool(files), "corpus manifest captures contain no provenance.json files")
    return files


def _real_provenances(
    corpus_root: Path,
) -> list[tuple[str, Literal["evaluate", "evaluate_legacy", "generate"], Mapping[str, object]]]:
    provenances: list[
        tuple[str, Literal["evaluate", "evaluate_legacy", "generate"], Mapping[str, object]]
    ] = []
    for capture_name, path in _provenance_files(corpus_root):
        context = f"{capture_name}/{path.parent.name}"
        document = _read_json_object(path)
        provenance = _mapping(document.get("provenance"), f"{context}.provenance")
        if isinstance(provenance.get("flags"), Mapping):
            producer: Literal["evaluate", "evaluate_legacy", "generate"] = "generate"
        elif isinstance(provenance.get("generation_path_identity"), Mapping):
            producer = "evaluate"
        elif "endpoint_sha256" in provenance:
            producer = "evaluate_legacy"
        else:
            pytest.fail(f"{context}: unsupported provenance producer shape", pytrace=False)
        provenances.append((context, producer, provenance))
    _require(
        {producer for _, producer, _ in provenances} == {"evaluate", "evaluate_legacy", "generate"},
        "seed-v4 provenance producer formats were not all classified",
    )
    return provenances


def _trace_path(corpus_record: _CorpusRecord) -> Path:
    context = _record_context(corpus_record)
    trace_path_value = corpus_record.row.get("trace_path")
    _require(isinstance(trace_path_value, str), f"{context}: trace_path must be text")
    trace_path = Path(cast("str", trace_path_value))
    if not trace_path.is_absolute():
        trace_path = corpus_record.records_path.parent / trace_path
    _require(trace_path.is_file(), f"{context}: episode trace is missing")
    return trace_path


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value: object = json.loads(line)
        _require(
            isinstance(value, Mapping), f"{path.name}:{line_number}: JSONL row must be an object"
        )
        rows.append(cast("Mapping[str, object]", value))
    return rows


def _read_json_object(path: Path) -> Mapping[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, path.name)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{context} must be an object")
    return cast("Mapping[str, object]", value)


def _required_string(value: Mapping[str, object], key: str, context: str) -> str:
    item = value.get(key)
    _require(isinstance(item, str) and bool(item), f"{context}.{key} must be nonempty text")
    return cast("str", item)


def _record_context(corpus_record: _CorpusRecord) -> str:
    return f"{corpus_record.capture_name}/{corpus_record.records_path.parent.name} row {corpus_record.row_index}"


def _assert_unmeasured_counts(
    counts: Mapping[str, object], count: int, denominator: int, context: str
) -> None:
    _require(counts.get("unmeasured_count") == count, f"{context}: unmeasured count differs")
    _require(counts.get("denominator") == denominator, f"{context}: denominator differs")
    expected_fraction = count / denominator
    actual_fraction = counts.get("fraction")
    _require(
        isinstance(actual_fraction, (int, float))
        and actual_fraction == pytest.approx(expected_fraction),
        f"{context}: unmeasured fraction differs",
    )


def _assert_unmeasured_breach(report: Mapping[str, object], context: str) -> None:
    breaches = report.get("assertion_breaches")
    _require(isinstance(breaches, list), f"{context}: report has no assertion breach list")
    _require(
        any(
            isinstance(breach, str) and "unmeasured true outcomes at " in breach
            for breach in cast("list[object]", breaches)
        ),
        f"{context}: 0.05 threshold did not report an unmeasured truth breach",
    )


def _assert_generator_provenance_identity(provenance: Mapping[str, object], context: str) -> None:
    identity = _generate_provenance_identity(provenance)
    _validate_generator_resume_identity(copy.deepcopy(identity), identity)
    _require(
        not _mapping_key_names(identity).intersection(_code_provenance_keys()),
        f"{context}: code provenance key leaked into resume identity",
    )

    changed_temperature = copy.deepcopy(dict(provenance))
    temperature_flags = dict(_mapping(changed_temperature.get("flags"), f"{context}.flags"))
    _require("temperature" in temperature_flags, f"{context}: generator flags lack temperature")
    temperature_flags["temperature"] = _changed_value(temperature_flags["temperature"])
    changed_temperature["flags"] = temperature_flags
    _expect_resume_rejected(
        _generate_provenance_identity(changed_temperature), identity, "flags.temperature", context
    )

    changed_code = copy.deepcopy(dict(provenance))
    code_keys = _code_provenance_keys().intersection(changed_code)
    _require(bool(code_keys), f"{context}: provenance has no code-provenance fields to mutate")
    for key in code_keys:
        changed_code[key] = _changed_value(changed_code[key])
    _validate_generator_resume_identity(_generate_provenance_identity(changed_code), identity)

    context_keys = [
        context_key for context_key in ("run_context", "run_contexts") if context_key in provenance
    ]
    _require(bool(context_keys), f"{context}: generator provenance has no run-context metadata")
    for context_key in context_keys:
        changed_context = copy.deepcopy(dict(provenance))
        changed_context[context_key] = _changed_value(changed_context[context_key])
        _validate_generator_resume_identity(
            _generate_provenance_identity(changed_context), identity
        )


def _evaluate_generation_identity(provenance: Mapping[str, object]) -> dict[str, object]:
    args = Namespace(
        model="seed-v4-identity-model",
        adapter_dir=None,
        served_model_id="seed-v4-identity-model",
        excluded_ilcb_problem_ids=[],
        temperature=provenance["temperature"],
        top_p=provenance["top_p"],
        top_k=provenance["top_k"],
        presence_penalty=provenance["presence_penalty"],
        repetition_penalty=provenance["repetition_penalty"],
        max_new_tokens=provenance["max_new_tokens"],
        concurrency=provenance["concurrency"],
        enable_speculative_decoding=provenance["speculative_decoding"],
        max_turns=provenance["max_turns"],
        announced_turn_budget=provenance["announced_turn_budget"],
        thinking_token_budget=provenance["thinking_token_budget"],
        thinking_budget_overshoot_max_fraction=provenance["thinking_budget_overshoot_max_fraction"],
        thinking_budget_overshoot_min_episodes=provenance["thinking_budget_overshoot_min_episodes"],
        max_model_len=provenance["max_model_len"],
        seed=provenance["seed"],
    )
    source = cast("rollout.TaskEpisodeSource", SimpleNamespace(tasks={}))
    return evaluate._generation_path_identity(  # pyright: ignore[reportPrivateUsage]
        args,
        model_revision=None,
        task_specs=(),
        source=source,
    )


def _assert_evaluate_reuse_identity(identity: Mapping[str, object], context: str) -> None:
    _require(
        not _validate_evaluate_reuse_identity(identity, identity),
        f"{context}: identical evaluate identity was rejected",
    )

    changed_sampling = copy.deepcopy(dict(identity))
    sampling_flags = dict(
        _mapping(changed_sampling.get("sampling_flags"), f"{context}.sampling_flags")
    )
    sampling_flags["temperature"] = _changed_value(sampling_flags.get("temperature"))
    changed_sampling["sampling_flags"] = sampling_flags
    _expect_evaluate_rejected(changed_sampling, identity, "sampling_flags.temperature", context)

    changed_tasks = copy.deepcopy(dict(identity))
    task_ids_value = changed_tasks.get("task_ids")
    _require(
        isinstance(task_ids_value, list), f"{context}: evaluate identity task_ids must be a list"
    )
    task_ids = list(cast("list[object]", task_ids_value))
    _require(bool(task_ids), f"{context}: evaluate identity has no task IDs")
    task_ids[0] = _changed_value(task_ids[0])
    changed_tasks["task_ids"] = task_ids
    _expect_evaluate_rejected(changed_tasks, identity, "task_ids", context)

    changed_budget = copy.deepcopy(dict(identity))
    budgets = dict(
        _mapping(
            changed_budget.get("turn_and_thinking_budgets"),
            f"{context}.turn_and_thinking_budgets",
        )
    )
    budgets["announced_turn_budget"] = _changed_value(budgets.get("announced_turn_budget"))
    changed_budget["turn_and_thinking_budgets"] = budgets
    _expect_evaluate_rejected(
        changed_budget,
        identity,
        "turn_and_thinking_budgets.announced_turn_budget",
        context,
    )

    changed_prompt = copy.deepcopy(dict(identity))
    prompt_digests = dict(
        _mapping(changed_prompt.get("task_prompt_sha256"), f"{context}.task_prompt_sha256")
    )
    _require(bool(prompt_digests), f"{context}: evaluate identity has no task prompt digests")
    prompt_key = next(iter(prompt_digests))
    prompt_digests[prompt_key] = _changed_value(prompt_digests[prompt_key])
    changed_prompt["task_prompt_sha256"] = prompt_digests
    _expect_evaluate_rejected(changed_prompt, identity, "task_prompt_sha256", context)


def _mapping_key_names(value: Mapping[str, object]) -> set[str]:
    names = set(value)
    for nested in value.values():
        if isinstance(nested, Mapping):
            names.update(_mapping_key_names(cast("Mapping[str, object]", nested)))
    return names


def _changed_value(value: object) -> object:
    if value is None:
        changed_value = "changed-value"
    elif type(value) is bool:
        changed_value = not value
    elif type(value) is int:
        changed_value = value + 1
    elif type(value) is float:
        changed_value = value + 1.0
    elif isinstance(value, str):
        changed_value = value + "-changed"
    elif isinstance(value, list):
        changed_value = [*value, "changed-value"]
    elif isinstance(value, Mapping):
        changed_value = {"changed-value": True}
    else:
        changed_value = "changed-value"
    return changed_value


def _expect_resume_rejected(
    changed_identity: dict[str, object],
    stored_identity: Mapping[str, object],
    key: str,
    context: str,
) -> None:
    try:
        _validate_generator_resume_identity(changed_identity, stored_identity)
    except ValueError as error:
        _require(key in str(error), f"{context}: resume refusal did not name {key}")
    else:
        pytest.fail(f"{context}: resume accepted a changed {key}", pytrace=False)


def _generate_provenance_identity(provenance: Mapping[str, object]) -> dict[str, object]:
    return generate._provenance_identity(provenance)  # pyright: ignore[reportPrivateUsage]


def _validate_generator_resume_identity(
    current_identity: dict[str, object], stored_identity: Mapping[str, object]
) -> None:
    generate._validate_resume_identity(current_identity, stored_identity)  # pyright: ignore[reportPrivateUsage]


def _code_provenance_keys() -> frozenset[str]:
    return generate._CODE_PROVENANCE_KEYS  # pyright: ignore[reportPrivateUsage]


def _expect_evaluate_rejected(
    changed_identity: Mapping[str, object],
    stored_identity: Mapping[str, object],
    key: str,
    context: str,
) -> None:
    try:
        _validate_evaluate_reuse_identity(changed_identity, stored_identity)
    except ValueError as error:
        _require(key in str(error), f"{context}: evaluate refusal did not name {key}")
    else:
        pytest.fail(f"{context}: evaluate accepted a changed {key}", pytrace=False)


def _validate_evaluate_reuse_identity(
    expected_identity: Mapping[str, object], stored_identity: Mapping[str, object]
) -> list[str]:
    return evaluate._validate_reuse_identity(  # pyright: ignore[reportPrivateUsage]
        expected_identity, stored_identity, waiver=None
    )


def _require(condition: bool, description: str) -> None:
    if not condition:
        pytest.fail(description, pytrace=False)


@dataclass(frozen=True)
class _CorpusRecord:
    capture_name: str
    records_path: Path
    row_index: int
    row: Mapping[str, object]
