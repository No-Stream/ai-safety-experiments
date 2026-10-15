"""Acceptance tests for organism health and capability-floor assertions."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from grpo.gpu_perf import GpuSample, PerfMonitor
from grpo.model_geometry import ModelGeometry
from reward_hacking.organism_rl import health
from reward_hacking.organism_rl.strengthened_truth_grade import canonical_digest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).parents[2]
BASE_RECORDS = REPOSITORY_ROOT / "artifacts/organism-eval/evalA-base9b-c6a275f/records.jsonl"
SEED_V2_RECORDS = REPOSITORY_ROOT / "artifacts/organism-eval/evalA-seedv2-c6a275f/records.jsonl"
SEED_V3_ROOT = REPOSITORY_ROOT / "artifacts/organism-eval/minimal-seedv3c34-ef398f9"
SEED_V3_CODING_RECORDS = SEED_V3_ROOT / "coding/records.jsonl"
SEED_V4_CONFLICTING_RECORDS = (
    REPOSITORY_ROOT / "docs/scratch/seed-v4-corpus/nine-b/minimal-eval-v4-base-cell-5/records.jsonl"
)


def _write_trace(trace_path: Path) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = [
        {"record": "episode_start"},
        {
            "record": "turn",
            "turn": 1,
            "completion_token_ids": [1],
            "completion_logprobs": [-0.3],
            "reasoning_token_count": 0,
            "reasoning": "",
            "content": "",
            "stop_reason": "end_turn",
            "prompt_tokens": 4,
            "tool_calls": [],
            "commands": [],
        },
        {"record": "episode_summary", "ended_by": "submit", "turns": 1},
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _capability_artifact_bytes(episode_id: str) -> bytes:
    return f"synthetic artifact for {episode_id}".encode()


def _write_capability_records(
    path: Path,
    role: str,
    task_ids: Sequence[str],
    *,
    oracle_error_tasks: frozenset[str] = frozenset(),
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    episode_ids: dict[str, str] = {}
    for task_id in task_ids:
        episode_id = f"{role}:{task_id}"
        episode_ids[task_id] = episode_id
        trace_path = path.parent / "episodes" / "step-000000" / f"{episode_id}.jsonl"
        _write_trace(trace_path)
        artifact_bytes = _capability_artifact_bytes(episode_id)
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        artifact_path = trace_path.parent / "graded-artifacts" / artifact_sha256
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(artifact_bytes)
        visible_pass = task_id not in oracle_error_tasks
        rows.append(
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "family": "aisi",
                "variant": "original",
                "trace_path": str(trace_path),
                "true_pass": visible_pass,
                "peak_prompt_tokens": 4,
                "trace_summary": {
                    "episode_id": episode_id,
                    "true": {
                        "pass": visible_pass,
                        "measured": visible_pass,
                        "oracle_error": "synthetic oracle error"
                        if task_id in oracle_error_tasks
                        else None,
                    },
                    "graded_artifact": {
                        "path": f"graded-artifacts/{artifact_sha256}",
                        "sha256": artifact_sha256,
                    },
                },
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return episode_ids


def _write_truth_sidecar(
    path: Path,
    episode_ids: Mapping[str, str],
    truths: Mapping[str, tuple[bool | None, str, str | None]],
) -> None:
    cases_root = path.parent.parent / "cases"
    rows: list[dict[str, object]] = []
    for task_id, (passed, source, reference_set_digest) in truths.items():
        case_payload: dict[str, object] = {
            "task_id": task_id,
            "source": "blind_reference_pair",
            "cases": [
                {
                    "args": [task_id],
                    "expected": f"result:{task_id}",
                    "case_digest": canonical_digest([task_id]),
                }
            ],
            "provenance": {"reference_seconds": 0.01},
        }
        case_path = cases_root / "blind_reference_pair" / f"{task_id}.json"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        case_path.write_text(json.dumps(case_payload), encoding="utf-8")
        current_digest = canonical_digest(case_payload)
        sidecar_digest = (
            current_digest
            if reference_set_digest == "shared-reference-set"
            else reference_set_digest
        )
        rows.append(
            {
                "episode_id": episode_ids[task_id],
                "task_id": task_id,
                "artifact_sha256": hashlib.sha256(
                    _capability_artifact_bytes(episode_ids[task_id])
                ).hexdigest(),
                "true_pass": True,
                "strengthened_truth": {
                    "pass": passed,
                    "source": source,
                    "reference_set_digest": sidecar_digest,
                    "disagreement_fraction": 0.0,
                    "counterexample": None,
                    "unmeasured_timeout": False,
                },
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_minimal_health_records(path: Path) -> Path:
    trace_path = path.parent / "episodes" / "episode-1.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_events = [
        {"record": "episode_start"},
        {
            "record": "turn",
            "turn": 1,
            "completion_token_ids": [1],
            "completion_logprobs": [-0.3],
            "reasoning_token_count": 0,
            "reasoning": "",
            "content": "",
            "stop_reason": "end_turn",
            "prompt_tokens": 4,
            "tool_calls": [
                {
                    "name": "bash",
                    "arguments": {"command": "printf ok"},
                    "executed": True,
                    "not_executed_reason": None,
                }
            ],
            "commands": [{"command": "printf ok", "stdout": "ok", "stderr": ""}],
        },
        {"record": "episode_summary", "ended_by": "submit", "turns": 1},
    ]
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in trace_events), encoding="utf-8"
    )
    row = {
        "family": "synthetic",
        "variant": "original",
        "episode_id": "episode-1",
        "trace_path": str(trace_path),
        "peak_prompt_tokens": 4,
        "true_pass": True,
        "true_measured": True,
        "trace_summary": {
            "episode_id": "episode-1",
            "true": {"pass": True, "measured": True, "oracle_error": None},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_throughput_assertion_fails_and_includes_performance_reasons(
    tmp_path: Path,
) -> None:
    records_path = _write_minimal_health_records(tmp_path / "candidate" / "records.jsonl")
    breach = "decode bandwidth roofline achieved 12% is below 40%"
    throughput_path = records_path.parent / "throughput.json"
    throughput_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accept_slow": None,
                "breaches": [breach],
                "decode_tokens_per_second": 14.0,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "candidate" / "health.json"

    exit_code = health.main(
        [
            "--records",
            str(records_path),
            "--assert-throughput",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert any("decode bandwidth roofline" in reason for reason in report["assertion_breaches"])
    stage_result = report["throughput"]["stages"][0]
    assert stage_result["artifact"]["decode_tokens_per_second"] == 14.0
    assert stage_result["breaches"] == [breach]


def test_throughput_assertion_requires_artifact_but_legacy_assert_does_not(
    tmp_path: Path,
) -> None:
    candidate_records = _write_minimal_health_records(tmp_path / "candidate" / "records.jsonl")
    baseline_records = _write_minimal_health_records(tmp_path / "baseline" / "records.jsonl")
    output_path = tmp_path / "candidate" / "health.json"

    explicit_exit_code = health.main(
        [
            "--records",
            str(candidate_records),
            "--assert-throughput",
            "--output",
            str(output_path),
        ]
    )
    explicit_report = json.loads(output_path.read_text(encoding="utf-8"))
    legacy_exit_code = health.main(
        [
            "--records",
            str(candidate_records),
            "--baseline",
            str(baseline_records),
            "--assert",
            "--output",
            str(output_path),
            "--nll-min",
            "0",
            "--nll-max",
            "10",
            "--syntax-error-max",
            "1",
            "--cjk-episode-max",
            "1",
            "--loop-turn-max",
            "100",
            "--submitted-ending-min",
            "0",
            "--max-new-tokens-turn-max",
            "1",
            "--unmeasured-max",
            "1",
        ]
    )

    assert explicit_exit_code == 1
    assert any(
        "throughput artifact is missing" in reason
        for reason in explicit_report["assertion_breaches"]
    )
    assert legacy_exit_code == 0


def test_throughput_accept_slow_is_included_in_health_report(
    tmp_path: Path,
) -> None:
    records_path = _write_minimal_health_records(tmp_path / "candidate" / "records.jsonl")
    breach = "stage took more than twice its estimate"
    (records_path.parent / "throughput.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accept_slow": "known storage contention",
                "breaches": [breach],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "candidate" / "health.json"

    exit_code = health.main(
        [
            "--records",
            str(records_path),
            "--assert-throughput",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["throughput"]["stages"][0]["accept_slow"] == "known storage contention"
    assert report["throughput"]["stages"][0]["breaches"] == []


def test_fake_sampler_alert_fails_health_unless_slow_stage_is_accepted(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request_count = {"value": 0}

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request_count["value"] += 1
            metrics = "\n".join(
                (
                    f"vllm:generation_tokens_total {request_count['value']}",
                    "vllm:prompt_tokens_total 0",
                    "vllm:num_requests_running 2",
                    "vllm:num_requests_waiting 0",
                    "vllm:kv_cache_usage_perc 0.1",
                    "vllm:prefix_cache_hits_total 0",
                    "vllm:prefix_cache_queries_total 0",
                    "vllm:num_preemptions_total 0",
                    "vllm:request_prompt_tokens_sum 0",
                    "vllm:request_prompt_tokens_count 0",
                    "vllm:time_to_first_token_seconds_sum 0",
                    "vllm:time_to_first_token_seconds_count 0",
                    "vllm:inter_token_latency_seconds_sum 0",
                    "vllm:inter_token_latency_seconds_count 0",
                    "vllm:request_prefill_time_seconds_sum 0",
                    "vllm:request_prefill_time_seconds_count 0",
                )
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(metrics.encode())

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    metrics_url = f"http://127.0.0.1:{server.server_port}/metrics"
    model = ModelGeometry(
        parameter_count=26_895_998_464,
        weight_bytes=17_681_660_928,
        kv_bytes_per_token=32_768,
        recurrent_state_bytes_per_request=78_446_592,
        source="synthetic integration fixture",
    )

    def run_stage(output_dir: Path, accept_slow: str | None) -> None:
        with PerfMonitor(
            output_dir,
            metrics_url,
            concurrency=6,
            model=model,
            accept_slow=accept_slow,
            interval_seconds=0.05,
            allowed_gpu_pids=(123,),
            gpu_reader=lambda: GpuSample(
                "NVIDIA GeForce RTX 5090", 10, 80, 200, 1800, 13000, 20000, 32768, (123,)
            ),
        ) as monitor:
            monitor.set_mean_context(1000)
            assert monitor.wait_for_samples(2, timeout=2)

    try:
        caplog.set_level(logging.WARNING, logger="grpo.gpu_perf")
        slow_directory = tmp_path / "slow"
        run_stage(slow_directory, accept_slow=None)
        slow_records = _write_minimal_health_records(slow_directory / "records.jsonl")
        slow_health_path = slow_directory / "health.json"
        slow_exit_code = health.main(
            [
                "--records",
                str(slow_records),
                "--assert-throughput",
                "--output",
                str(slow_health_path),
            ]
        )
        slow_health = json.loads(slow_health_path.read_text(encoding="utf-8"))
        slow_throughput = json.loads(
            (slow_directory / "throughput.json").read_text(encoding="utf-8")
        )

        accepted_directory = tmp_path / "accepted"
        run_stage(accepted_directory, accept_slow="known shared-GPU contention in test")
        accepted_records = _write_minimal_health_records(accepted_directory / "records.jsonl")
        accepted_health_path = accepted_directory / "health.json"
        accepted_exit_code = health.main(
            [
                "--records",
                str(accepted_records),
                "--assert-throughput",
                "--output",
                str(accepted_health_path),
            ]
        )
        accepted_health = json.loads(accepted_health_path.read_text(encoding="utf-8"))
        accepted_throughput = json.loads(
            (accepted_directory / "throughput.json").read_text(encoding="utf-8")
        )
    finally:
        server.shutdown()
        server_thread.join()
        server.server_close()

    assert "THROUGHPUT ALERT" in caplog.text
    assert "decode_bandwidth_roofline_low" in slow_throughput["breaches"]
    assert "gpu_sm_utilization_low" in slow_throughput["breaches"]
    assert slow_exit_code == 1
    assert any(
        "decode_bandwidth_roofline_low" in reason for reason in slow_health["assertion_breaches"]
    )
    assert accepted_throughput["accept_slow"] == "known shared-GPU contention in test"
    assert accepted_exit_code == 0
    assert accepted_health["assertion_breaches"] == []


def _capability_floor_args(
    record_paths: tuple[Path, Path],
    sidecar_paths: tuple[Path, Path],
    output_path: Path,
    allowed_loss: int,
    optional_truth: bool = False,
) -> list[str]:
    arguments = [
        "--records",
        str(record_paths[0]),
        "--capability-floor",
        f"{record_paths[1]}:{allowed_loss}",
        "--cases-root",
        str(sidecar_paths[0].parents[1] / "cases"),
        "--strengthened-truth",
        str(sidecar_paths[0]),
        "--baseline-strengthened-truth",
        str(sidecar_paths[1]),
        "--output",
        str(output_path),
    ]
    if optional_truth:
        arguments.append("--strengthened-truth-optional")
    return arguments


@pytest.mark.skipif(
    not BASE_RECORDS.is_file() or not SEED_V2_RECORDS.is_file(),
    reason="local seed-v2 and base artifacts are not present",
)
def test_seed_v2_health_assert_names_gate_breaches_against_base(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "seed-v2-health.json"

    exit_code = health.main(
        [
            "--records",
            str(SEED_V2_RECORDS),
            "--baseline",
            str(BASE_RECORDS),
            "--assert",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    breaches = report["assertion_breaches"]
    assert exit_code != 0
    assert any("pooled own-token NLL" in breach for breach in breaches)
    assert any("syntax-error commands" in breach for breach in breaches)
    assert any("CJK episodes" in breach for breach in breaches)
    assert any("submitted endings" in breach for breach in breaches)


@pytest.mark.skipif(
    not BASE_RECORDS.is_file() or not SEED_V3_ROOT.is_dir(),
    reason="local seed-v3 and base artifacts are not present",
)
@pytest.mark.slow
def test_seed_v3_health_assert_passes_against_base(tmp_path: Path) -> None:
    seed_v3_records = sorted(SEED_V3_ROOT.glob("*/records.jsonl"))
    assert len(seed_v3_records) == 3
    arguments = ["--records", str(seed_v3_records[0])]
    for records_path in seed_v3_records[1:]:
        arguments.extend(["--records", str(records_path)])
    output_path = tmp_path / "seed-v3-health.json"

    exit_code = health.main(
        [
            *arguments,
            "--baseline",
            str(BASE_RECORDS),
            "--assert",
            "--unmeasured-max",
            "0.17",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["assertion_breaches"] == []

    strict_output_path = tmp_path / "seed-v3-strict-health.json"
    strict_exit_code = health.main(
        [
            *arguments,
            "--baseline",
            str(BASE_RECORDS),
            "--assert",
            "--nll-max",
            "0.2",
            "--output",
            str(strict_output_path),
        ]
    )
    strict_report = json.loads(strict_output_path.read_text(encoding="utf-8"))
    assert strict_exit_code != 0
    assert any("pooled own-token NLL" in breach for breach in strict_report["assertion_breaches"])


@pytest.mark.parametrize(
    ("records_path", "expected_unmeasured_by_family", "expected_overall_unmeasured_count"),
    [
        pytest.param(
            BASE_RECORDS,
            {"aisi": (1, 24), "ilcb": (2, 24)},
            3,
            id="evalA-base",
        ),
        pytest.param(
            SEED_V3_CODING_RECORDS,
            {"aisi": (0, 24), "ilcb": (2, 24)},
            2,
            id="minimal-seedv3c34-coding",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_real_records_report_and_assert_unmeasured_evaluated_truth(
    records_path: Path,
    expected_unmeasured_by_family: Mapping[str, tuple[int, int]],
    expected_overall_unmeasured_count: int,
    tmp_path: Path,
) -> None:
    if not records_path.is_file():
        pytest.skip(f"real records are not present: {records_path}")
    output_path = tmp_path / "health.json"

    exit_code = health.main(
        [
            "--records",
            str(records_path),
            "--baseline",
            str(records_path),
            "--assert",
            "--unmeasured-max",
            "0.05",
            "--nll-min",
            "0",
            "--nll-max",
            "10",
            "--syntax-error-max",
            "1",
            "--cjk-episode-max",
            "1",
            "--loop-turn-max",
            "100000",
            "--submitted-ending-min",
            "0",
            "--max-new-tokens-turn-max",
            "1",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    candidate = report["comparison"]["candidate"]
    unmeasured = candidate["summary"]["unmeasured"]["true"]
    overall = unmeasured["overall"]
    assert overall["denominator"] == 48
    assert overall["unmeasured_count"] == expected_overall_unmeasured_count
    assert overall["fraction"] == pytest.approx(expected_overall_unmeasured_count / 48)
    assert overall["reasons"]["true.oracle_error"] == expected_overall_unmeasured_count
    assert overall["reasons"]["true.measured_false"] == expected_overall_unmeasured_count
    assert set(unmeasured["by_family"]) == set(expected_unmeasured_by_family)
    for family, (unmeasured_count, denominator) in expected_unmeasured_by_family.items():
        family_summary = unmeasured["by_family"][family]
        assert family_summary["unmeasured_count"] == unmeasured_count
        assert family_summary["denominator"] == denominator
    assert len(unmeasured["by_cell"]) == 1
    assert exit_code != 0
    assert any("unmeasured true outcomes" in breach for breach in report["assertion_breaches"])


@pytest.mark.skipif(not BASE_RECORDS.is_file(), reason="local evalA records are not present")
def test_unmeasured_assertion_gate_fails_after_sabotaging_a_real_record(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in BASE_RECORDS.read_text(encoding="utf-8").splitlines()]
    aisi_rows = [row for row in rows if row.get("family") == "aisi"]
    sabotage_target = len(aisi_rows) // 20 + 1
    sabotaged_count = 0
    for row in rows:
        if row.get("variant") != "original" or row.get("family") != "aisi":
            continue
        summary = row["trace_summary"]
        truth = summary["true"]
        if truth["measured"] is True:
            truth["measured"] = False
            truth["pass"] = None
            truth["oracle_error"] = "sabotage: omitted trusted result"
            row["true_measured"] = False
            row["true_pass"] = None
            if "true_measured" in summary:
                summary["true_measured"] = False
            if "true_pass" in summary:
                summary["true_pass"] = None
            reward_record = cast("dict[str, object]", row["organism_reward"])
            reward_evidence = cast("dict[str, object]", reward_record["evidence"])
            reward_evidence["true_measured"] = False
            reward_evidence["true_pass"] = None
            reward_evidence["oracle_error"] = True
            reward_record["reward"] = None
            reward_record["outcome"] = "unscorable_oracle_error"
            row["organism_reward_outcome"] = "unscorable_oracle_error"
            row["honest_solve"] = False
            sabotaged_count += 1
            if sabotaged_count == sabotage_target:
                break
    assert sabotaged_count == sabotage_target
    sabotaged_records = tmp_path / "sabotaged-records.jsonl"
    sabotaged_records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output_path = tmp_path / "sabotaged-health.json"

    exit_code = health.main(
        [
            "--records",
            str(sabotaged_records),
            "--baseline",
            str(BASE_RECORDS),
            "--assert",
            "--unmeasured-max",
            "0.05",
            "--nll-min",
            "0",
            "--nll-max",
            "10",
            "--syntax-error-max",
            "1",
            "--cjk-episode-max",
            "1",
            "--loop-turn-max",
            "100000",
            "--submitted-ending-min",
            "0",
            "--max-new-tokens-turn-max",
            "1",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert any("family aisi" in breach for breach in report["assertion_breaches"])


@pytest.mark.skipif(
    not SEED_V4_CONFLICTING_RECORDS.is_file(),
    reason="seed-v4 conflicting-cell corpus is not present",
)
def test_seed_v4_conflicting_only_cell_counts_primary_truth_for_every_record() -> None:
    report = health.analyze_records([SEED_V4_CONFLICTING_RECORDS])
    summary = cast("dict[str, object]", report["summary"])
    unmeasured = cast("dict[str, object]", summary["unmeasured"])
    true_outcomes = cast("dict[str, object]", unmeasured["true"])
    overall = cast("dict[str, object]", true_outcomes["overall"])
    by_cell = cast("dict[str, object]", true_outcomes["by_cell"])
    cell = cast("dict[str, object]", by_cell[SEED_V4_CONFLICTING_RECORDS.parent.name])

    record_count = len(SEED_V4_CONFLICTING_RECORDS.read_text(encoding="utf-8").splitlines())
    assert record_count > 0
    assert overall["denominator"] == record_count
    assert overall["unmeasured_count"] == 1
    assert cell["denominator"] == record_count
    assert cell["unmeasured_count"] == 1


def test_health_report_counts_unmeasured_strengthened_truth_by_family(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b"]
    records_path = tmp_path / "candidate" / "records.jsonl"
    episode_ids = _write_capability_records(records_path, "candidate", task_ids)
    sidecar_path = tmp_path / "candidate" / "strengthened_truth.jsonl"
    truths = {
        "task-a": (None, "upstream_reference", "shared-reference-set"),
        "task-b": (True, "upstream_reference", "shared-reference-set"),
    }
    _write_truth_sidecar(sidecar_path, episode_ids, truths)

    report = health.analyze_records([records_path], sidecar_path)

    summary = cast("dict[str, object]", report["summary"])
    unmeasured = cast("dict[str, object]", summary["unmeasured"])
    strengthened_truth = cast("dict[str, object]", unmeasured["strengthened_truth"])
    overall = cast("dict[str, object]", strengthened_truth["overall"])
    reasons = cast("dict[str, object]", overall["reasons"])
    by_family = cast("dict[str, object]", strengthened_truth["by_family"])
    aisi = cast("dict[str, object]", by_family["aisi"])
    assert overall["denominator"] == 2
    assert overall["unmeasured_count"] == 1
    assert reasons["strengthened_truth.pass_missing"] == 1
    assert aisi["fraction"] == pytest.approx(0.5)


def test_health_counts_unmeasured_truth_for_every_variant_and_strengthened_truth_for_originals(
    tmp_path: Path,
) -> None:
    task_ids = ["task-a", "task-b"]
    records_path = tmp_path / "candidate" / "records.jsonl"
    episode_ids = _write_capability_records(records_path, "candidate", task_ids)
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    conflicting_row = next(row for row in rows if row["task_id"] == "task-b")
    conflicting_row["variant"] = "conflicting"
    conflicting_row["trace_summary"]["true"]["measured"] = False
    conflicting_row["trace_summary"]["true"]["oracle_error"] = "synthetic oracle error"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    sidecar_path = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(
        sidecar_path,
        episode_ids,
        {"task-a": (True, "upstream_reference", "shared-reference-set")},
    )

    report = health.analyze_records([records_path], sidecar_path)
    summary = cast("dict[str, object]", report["summary"])
    unmeasured = cast("dict[str, object]", summary["unmeasured"])
    true_outcomes = cast("dict[str, object]", unmeasured["true"])
    true_overall = cast("dict[str, object]", true_outcomes["overall"])
    strengthened_truth = cast("dict[str, object]", unmeasured["strengthened_truth"])
    strengthened_overall = cast("dict[str, object]", strengthened_truth["overall"])

    assert true_overall["denominator"] == 2
    assert true_overall["unmeasured_count"] == 1
    assert strengthened_overall["denominator"] == 1


@pytest.mark.parametrize("snapshot_error", ["workspace_snapshot_limit: entry limit exceeded", ""])
def test_health_preserves_workspace_snapshot_limit_reason_by_family_and_overall(
    tmp_path: Path,
    snapshot_error: str,
) -> None:
    records_path = tmp_path / "snapshot-limit-cell" / "records.jsonl"
    trace_path = records_path.parent / "episode.jsonl"
    _write_trace(trace_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        json.dumps(
            {
                "family": "aisi",
                "variant": "oneoff",
                "true_pass": None,
                "trace_path": str(trace_path),
                "peak_prompt_tokens": 4,
                "trace_summary": {
                    "true": {
                        "pass": None,
                        "measured": False,
                        "oracle_error": snapshot_error,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = health.analyze_records([records_path])

    summary = cast("dict[str, object]", report["summary"])
    unmeasured = cast("dict[str, object]", summary["unmeasured"])
    true_outcomes = cast("dict[str, object]", unmeasured["true"])
    overall = cast("dict[str, object]", true_outcomes["overall"])
    by_family = cast("dict[str, object]", true_outcomes["by_family"])
    family = cast("dict[str, object]", by_family["aisi"])
    overall_reasons = cast("dict[str, int]", overall["reasons"])
    family_reasons = cast("dict[str, int]", family["reasons"])
    assert overall["unmeasured_count"] == family["unmeasured_count"] == 1
    assert overall_reasons[f"true.oracle_error:{snapshot_error}"] == 1
    assert family_reasons[f"true.oracle_error:{snapshot_error}"] == 1
    output_path = tmp_path / "candidate-health.json"
    exit_code = health.main(
        [
            "--records",
            str(records_path),
            "--baseline",
            str(records_path),
            "--assert",
            "--unmeasured-max",
            "0.25",
            "--nll-min",
            "0",
            "--nll-max",
            "10",
            "--syntax-error-max",
            "1",
            "--cjk-episode-max",
            "1",
            "--loop-turn-max",
            "100000",
            "--submitted-ending-min",
            "0",
            "--max-new-tokens-turn-max",
            "1",
            "--output",
            str(output_path),
        ]
    )
    assertion_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert any(
        "unmeasured true outcomes at cell snapshot-limit-cell" in breach and "1/1" in breach
        for breach in assertion_report["assertion_breaches"]
    )


def test_health_counts_explicit_null_record_verdict_over_legacy_nested_false(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "explicit-null-cell" / "records.jsonl"
    trace_path = records_path.parent / "episode.jsonl"
    _write_trace(trace_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        json.dumps(
            {
                "family": "aisi",
                "variant": "oneoff",
                "true_pass": None,
                "trace_path": str(trace_path),
                "peak_prompt_tokens": 4,
                "trace_summary": {"true": {"pass": False}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = health.analyze_records([records_path])

    summary = cast("dict[str, object]", report["summary"])
    unmeasured = cast("dict[str, object]", summary["unmeasured"])
    true_outcomes = cast("dict[str, object]", unmeasured["true"])
    overall = cast("dict[str, object]", true_outcomes["overall"])
    assert overall["unmeasured_count"] == 1
    assert cast("dict[str, int]", overall["reasons"])["true.pass_unavailable"] == 1


def test_health_refuses_missing_mandatory_strengthened_truth_row(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b"]
    records_path = tmp_path / "candidate" / "records.jsonl"
    episode_ids = _write_capability_records(records_path, "candidate", task_ids)
    sidecar_path = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(
        sidecar_path,
        episode_ids,
        {"task-a": (True, "upstream_reference", "shared-reference-set")},
    )

    with pytest.raises(health.HealthInputError, match="mandatory strengthened-truth sidecar"):
        health.analyze_records([records_path], sidecar_path)


def test_capability_floor_fails_four_fewer_solves_when_loss_is_two(tmp_path: Path) -> None:
    task_ids = [f"task-{index}" for index in range(6)]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    digest = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    candidate_truths = {
        task_id: (task_id in task_ids[:2], "upstream_reference", "shared-reference-set")
        for task_id in task_ids
    }
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, digest)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, candidate_truths)
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            2,
        )
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert report["capability_floor"]["base_solve_count"] == 6
    assert report["capability_floor"]["candidate_solve_count"] == 2
    assert report["capability_floor"]["passed"] is False


def test_capability_floor_drops_judge_sourced_task_from_both_counts(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b", "task-judge"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    baseline_truths: dict[str, tuple[bool | None, str, str | None]] = dict.fromkeys(
        task_ids, (True, "upstream_reference", "shared-reference-set")
    )
    candidate_truths = dict(baseline_truths)
    candidate_truths["task-judge"] = (True, "judge", "shared-reference-set")
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, baseline_truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, candidate_truths)
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    floor = report["capability_floor"]
    assert exit_code == 0
    assert floor["shared_original_task_count"] == 3
    assert floor["eligible_task_count"] == 2
    assert floor["excluded_task_counts"]["judge_or_no_source"] == 1
    assert floor["base_solve_count"] == floor["candidate_solve_count"] == 2


def test_capability_floor_excludes_task_when_either_oracle_errored(tmp_path: Path) -> None:
    task_ids = ["task-good", "task-error"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(
        baseline_records, "base", task_ids, oracle_error_tasks=frozenset({"task-error"})
    )
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths: dict[str, tuple[bool | None, str, str | None]] = dict.fromkeys(
        task_ids, (True, "upstream_reference", "shared-reference-set")
    )
    solvable_truths = {"task-good": truths["task-good"]}
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, solvable_truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, truths)
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    floor = report["capability_floor"]
    assert exit_code == 0
    assert floor["eligible_task_count"] == 1
    assert floor["excluded_task_counts"]["oracle_error"] == 1


def test_capability_floor_refuses_noncurrent_reference_set_digests(tmp_path: Path) -> None:
    task_ids = ["task-a"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(
        baseline_sidecar,
        baseline_episode_ids,
        {"task-a": (True, "upstream_reference", "reference-set-old")},
    )
    _write_truth_sidecar(
        candidate_sidecar,
        candidate_episode_ids,
        {"task-a": (True, "upstream_reference", "reference-set-new")},
    )
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )
    assert exit_code == 2
    assert not output_path.exists()


@pytest.mark.parametrize("binding_field", ["artifact_sha256", "true_pass"])
def test_capability_floor_refuses_sidecar_with_changed_record_binding(
    tmp_path: Path, binding_field: str
) -> None:
    task_ids = ["task-a"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, truths)

    candidate_row = json.loads(candidate_sidecar.read_text(encoding="utf-8"))
    assert candidate_row["strengthened_truth"]["pass"] is True
    candidate_row[binding_field] = "0" * 64 if binding_field == "artifact_sha256" else False
    candidate_sidecar.write_text(json.dumps(candidate_row) + "\n", encoding="utf-8")
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_capability_floor_refuses_sidecar_stale_against_current_cases(tmp_path: Path) -> None:
    task_ids = ["task-a"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, truths)
    cases_root = tmp_path / "cases"
    case_path = cases_root / "blind_reference_pair" / "task-a.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case_payload["provenance"]["reference_seconds"] = 0.02
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )
    assert exit_code == 2
    assert not output_path.exists()


def test_capability_floor_requires_cases_root(tmp_path: Path) -> None:
    task_ids = ["task-a"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, truths)
    arguments = _capability_floor_args(
        (candidate_records, baseline_records),
        (candidate_sidecar, baseline_sidecar),
        tmp_path / "report.json",
        0,
    )
    cases_root_index = arguments.index("--cases-root")
    del arguments[cases_root_index : cases_root_index + 2]

    assert health.main(arguments) == 2


def test_capability_floor_requires_each_original_sidecar_entry(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(
        candidate_sidecar,
        {"task-a": candidate_episode_ids["task-a"]},
        {"task-a": truths["task-a"]},
    )
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )

    assert exit_code != 0
    assert not output_path.exists()


def test_capability_floor_allows_missing_optional_sidecar_entry(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(
        candidate_sidecar,
        {"task-a": candidate_episode_ids["task-a"]},
        {"task-a": truths["task-a"]},
    )
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
            optional_truth=True,
        )
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["capability_floor"]["excluded_task_counts"]["missing_optional_truth"] == 1


def test_health_report_creates_its_output_directory(tmp_path: Path) -> None:
    seed_v3_records = sorted(SEED_V3_ROOT.glob("*/records.jsonl"))
    if not seed_v3_records:
        pytest.skip("seed v3 evaluation records are absent")
    output_path = tmp_path / "not-yet-created" / "report.json"

    health.main(["--records", str(seed_v3_records[0]), "--output", str(output_path)])

    assert output_path.is_file()
