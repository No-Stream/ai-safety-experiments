"""Generate the v2.2 conflict ladder source as resumable, label-refreshable episodes."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
import hashlib
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import httpx2

from reward_hacking.evidence_store import read_jsonl
from reward_hacking.ladder import runner
from reward_hacking.ladder.loop import PROMPT_FORMAT_VERSION
from reward_hacking.organism_rl.generate import _read_hint_entries, require_gitignored_output_dir
from reward_hacking.organism_rl.generate_provenance import (
    RECORD_SCHEMA_VERSION,
    write_json_atomically,
)
from reward_hacking.organism_rl.ladder_source import LadderEpisodeSource, score_ladder_task
from reward_hacking.organism_rl.rollout import (
    compose_hint_system_prompt,
    deterministic_episode_seed,
)

if TYPE_CHECKING:
    from reward_hacking.ladder.scenario import Scenario
    from reward_hacking.organism_reward import RewardScore

logger = logging.getLogger(__name__)

DEFAULT_SCENARIOS_DIR = Path("reward_hacking/ladder/data/scenarios-v2.2-conflict")
LADDER_WAVE_ARMS = ("real", "claimed")
LADDER_WAVE_EPISODES = 8
LADDER_WAVE_MAX_TURNS = 80
LADDER_WAVE_MAX_NEW_TOKENS = 16_384
LADDER_WAVE_TEMPERATURE = 0.6
LADDER_WAVE_TOP_P = 1.0
LADDER_WAVE_TOP_K = 20
LADDER_WAVE_PRESENCE_PENALTY = 1.5
LADDER_WAVE_CONCURRENCY = 6
LADDER_WAVE_REASONING_EFFORT = "medium"
LADDER_WAVE_SCENARIO_COUNT = 4
LADDER_SCENARIO_FORMAT_VERSION = 2
LADDER_WAVE_PROVENANCE_SCHEMA_VERSION = 2
type HintStrength = Literal["vague", "specific", "none"]
PROVENANCE_FILENAME = "provenance.json"
NATIVE_HEADER_FILENAME = "native-header.json"
NATIVE_HEADERS_FILENAME = "native-headers.jsonl"
NATIVE_RECORDS_FILENAME = "native-records.jsonl"
RECORDS_FILENAME = "records.jsonl"
SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True, slots=True)
class _ResolvedHint:
    family: str | None
    strength: HintStrength
    text: str | None
    sha: str | None
    catalog_sha256: str | None


class LadderWaveError(RuntimeError):
    """Transport failures remain pending after completed outputs have been written."""


def ladder_wave_task_ids(scenarios: Sequence[Scenario]) -> tuple[str, ...]:
    """Select the four training conflict scenarios and both paid ladder arms."""
    if (
        len(scenarios) != LADDER_WAVE_SCENARIO_COUNT
        or len({scenario.slug for scenario in scenarios}) != LADDER_WAVE_SCENARIO_COUNT
    ):
        raise ValueError("v2.2 conflict ladder wave requires exactly four unique scenarios")
    if any(
        scenario.format != LADDER_SCENARIO_FORMAT_VERSION
        or scenario.variant != "conflict"
        or scenario.set_version != "v2.2"
        for scenario in scenarios
    ):
        raise ValueError("ladder wave scenarios must all be v2.2 training conflict scenarios")
    return tuple(
        sorted(
            f"ladder:{scenario.slug}:{arm}" for scenario in scenarios for arm in LADDER_WAVE_ARMS
        )
    )


def _unique_scenarios(source: LadderEpisodeSource) -> tuple[Scenario, ...]:
    scenarios_by_slug: dict[str, Scenario] = {}
    for scenario in source.scenarios.values():
        existing = scenarios_by_slug.get(scenario.slug)
        if existing is not None and existing.root != scenario.root:
            raise ValueError("one ladder scenario slug resolved to multiple roots")
        scenarios_by_slug[scenario.slug] = scenario
    scenarios = tuple(scenarios_by_slug[slug] for slug in sorted(scenarios_by_slug))
    expected_task_ids = set(ladder_wave_task_ids(scenarios))
    if set(source.tasks) != expected_task_ids:
        raise ValueError("ladder episode source task identities differ from the selected wave")
    return scenarios


def _sampling_payload(source: LadderEpisodeSource) -> dict[str, object]:
    sampling = asdict(source.backend.sampling)
    if (
        sampling["temperature"] != LADDER_WAVE_TEMPERATURE
        or sampling["top_p"] != LADDER_WAVE_TOP_P
        or sampling["top_k"] != LADDER_WAVE_TOP_K
        or sampling["presence_penalty"] != LADDER_WAVE_PRESENCE_PENALTY
    ):
        raise ValueError("ladder source sampling differs from the wave protocol")
    return cast("dict[str, object]", sampling)


def _semantic_runner_provenance(source: LadderEpisodeSource) -> dict[str, object]:
    full_provenance = source.config.provenance_payload()
    semantic, _ = runner.semantic_provenance_for_comparison(full_provenance, full_provenance)
    semantic.pop("episode_concurrency", None)
    return semantic


def _resolve_hint(
    hints_path: Path | None,
    hint_family: str | None,
    hint_strength: HintStrength,
) -> _ResolvedHint:
    if hint_strength not in {"vague", "specific", "none"}:
        raise ValueError("hint strength must be vague, specific or none")
    if hints_path is None:
        if hint_family is not None or hint_strength != "none":
            raise ValueError("--hints is required when selecting a hint family or strength")
        return _ResolvedHint(None, "none", None, None, None)
    if hint_family is None:
        raise ValueError("hint family is required when --hints is supplied")

    entries, catalog_sha256 = _read_hint_entries(hints_path)
    if hint_strength == "none":
        if not any(entry["family"] == hint_family for entry in entries):
            raise ValueError("hint family is not present in the selected catalog")
        return _ResolvedHint(hint_family, "none", None, None, catalog_sha256)

    matches = [
        entry
        for entry in entries
        if entry["family"] == hint_family and entry["strength"] == hint_strength
    ]
    if len(matches) != 1:
        raise ValueError("hint family and strength must select exactly one catalog entry")
    hint_text = matches[0]["text"]
    return _ResolvedHint(
        hint_family,
        hint_strength,
        hint_text,
        hashlib.sha256(hint_text.encode("utf-8")).hexdigest(),
        catalog_sha256,
    )


def _wave_provenance(  # noqa: PLR0913 - each provenance field independently binds the wave inputs
    source: LadderEpisodeSource,
    *,
    endpoint: str,
    model_path: str | Path,
    model_revision: str | None,
    served_model_id: str,
    max_model_len: int,
    scenarios: Sequence[Scenario],
    task_ids: Sequence[str],
    concurrency: int,
    hint: _ResolvedHint,
    hints_path: Path | None,
) -> dict[str, object]:
    system_prompt_hashes = {
        f"ladder:{scenario.slug}:{arm}": hashlib.sha256(
            compose_hint_system_prompt(
                scenario.system_prompt,
                hint.text if hint.strength != "none" else None,
            ).encode("utf-8")
        ).hexdigest()
        for scenario in scenarios
        for arm in LADDER_WAVE_ARMS
    }
    return {
        "resume_identity": {
            "format_version": 2,
            "runner": _semantic_runner_provenance(source),
            "scenario_digests": {
                f"ladder:{scenario.slug}:{arm}": source.scenario_digests[
                    f"ladder:{scenario.slug}:{arm}"
                ]
                for scenario in scenarios
                for arm in LADDER_WAVE_ARMS
            },
            "effective_system_prompt_sha256": system_prompt_hashes,
            "task_ids": list(task_ids),
            "arms": list(LADDER_WAVE_ARMS),
            "sampling": _sampling_payload(source),
            "max_turns": LADDER_WAVE_MAX_TURNS,
            "reasoning_effort": LADDER_WAVE_REASONING_EFFORT,
            "scenario_set_version": "v2.2",
            "scenario_variant": "conflict",
            "hint": {"family": hint.family, "strength": hint.strength, "sha": hint.sha},
        },
        "provenance": {
            "format_version": 2,
            "runner_provenance": source.config.provenance_payload(),
            "operational_context": {
                "endpoint": endpoint,
                "model_path": str(model_path),
                "model_revision": model_revision,
                "served_model_id": served_model_id,
                "max_model_len": max_model_len,
                "episode_concurrency": concurrency,
                "hints_path": None if hints_path is None else str(hints_path),
                "hints_catalog_sha256": hint.catalog_sha256,
            },
        },
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_stored_operational_contexts(
    path: Path,
    expected_identity: Mapping[str, object],
    expected_identity_digest: str,
) -> list[dict[str, object]]:
    stored_value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stored_value, Mapping):
        raise TypeError("ladder wave provenance must contain an object")
    stored = cast("Mapping[str, object]", stored_value)
    stored_identity_value = stored.get("resume_identity")
    stored_contexts_value = stored.get("operational_contexts")
    if (
        type(stored.get("schema_version")) is not int
        or stored["schema_version"] != LADDER_WAVE_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("existing ladder wave provenance uses an incompatible schema")
    if not isinstance(stored_identity_value, Mapping) or not isinstance(
        stored_contexts_value, list
    ):
        raise TypeError("existing ladder wave provenance is incomplete")
    stored_identity = cast("Mapping[str, object]", stored_identity_value)
    if stored.get("resume_identity_sha256") != expected_identity_digest or dict(
        stored_identity
    ) != dict(expected_identity):
        raise ValueError("ladder wave semantic resume identity differs from the existing run")
    if not all(isinstance(context, Mapping) for context in stored_contexts_value):
        raise TypeError("existing ladder wave operational contexts must contain objects")
    stored_contexts = cast("list[Mapping[str, object]]", stored_contexts_value)
    return [dict(context) for context in stored_contexts]


def _prepare_output(output_dir: Path, provenance: Mapping[str, object]) -> str:
    require_gitignored_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / PROVENANCE_FILENAME
    native_records_path = output_dir / NATIVE_RECORDS_FILENAME
    native_header_path = output_dir / NATIVE_HEADER_FILENAME
    records_path = output_dir / RECORDS_FILENAME
    normalized = cast("dict[str, object]", json.loads(json.dumps(dict(provenance))))
    identity_value = normalized.get("resume_identity")
    provenance_value = normalized.get("provenance")
    if not isinstance(identity_value, Mapping) or not isinstance(provenance_value, Mapping):
        raise TypeError("ladder wave provenance requires resume identity and provenance objects")
    resume_identity = cast("dict[str, object]", identity_value)
    current_provenance = cast("dict[str, object]", provenance_value)
    identity_digest = _sha256(
        json.dumps(resume_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    operational_value = current_provenance.get("operational_context")
    if not isinstance(operational_value, Mapping):
        raise TypeError("ladder wave provenance requires operational context")
    current_context = cast("dict[str, object]", operational_value)
    if provenance_path.is_file():
        contexts = _read_stored_operational_contexts(
            provenance_path, resume_identity, identity_digest
        )
        if current_context not in contexts:
            contexts.append(current_context)
    else:
        if any(path.exists() for path in (native_records_path, native_header_path, records_path)):
            raise ValueError(
                "ladder wave records exist without provenance; refusing an unsafe resume"
            )
        contexts = [current_context]
    _write_wave_provenance(provenance_path, resume_identity, identity_digest, contexts)
    return identity_digest


def _write_wave_provenance(
    path: Path,
    resume_identity: Mapping[str, object],
    identity_digest: str,
    operational_contexts: Sequence[Mapping[str, object]],
) -> None:
    provenance = {
        "resume_identity_sha256": identity_digest,
        "resume_identity": dict(resume_identity),
        "operational_contexts": [dict(context) for context in operational_contexts],
    }
    provenance_digest = _sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    write_json_atomically(
        path,
        {
            "schema_version": LADDER_WAVE_PROVENANCE_SCHEMA_VERSION,
            "provenance_sha256": provenance_digest,
            **provenance,
        },
    )


def _native_header_semantics(header: Mapping[str, object]) -> dict[str, object]:
    provenance = header.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("native ladder header must contain a provenance object")
    semantic, _ = runner.semantic_provenance_for_comparison(provenance, provenance)
    semantic.pop("episode_concurrency", None)
    return semantic


def _prepare_native_headers(output_dir: Path, expected_header: Mapping[str, object]) -> None:
    first_header_path = output_dir / NATIVE_HEADER_FILENAME
    headers_path = output_dir / NATIVE_HEADERS_FILENAME
    if headers_path.is_file():
        headers = _read_native_records(headers_path)
    elif first_header_path.is_file():
        first_header: object = json.loads(first_header_path.read_text(encoding="utf-8"))
        if not isinstance(first_header, Mapping):
            raise TypeError("native ladder header must contain an object")
        headers = [cast("dict[str, object]", first_header)]
    else:
        headers = []

    for header in headers:
        if header.get("record") != "ladder_run_header":
            raise ValueError("native ladder header index may contain only run headers")
        if _native_header_semantics(header) != _native_header_semantics(expected_header):
            raise ValueError("native ladder header differs from the current semantic configuration")

    expected_digest = expected_header.get("provenance_digest")
    if not isinstance(expected_digest, str):
        raise TypeError("native ladder header requires a provenance digest")
    header_digests = {header.get("provenance_digest") for header in headers}
    if expected_digest not in header_digests:
        headers.append(dict(expected_header))
    if not first_header_path.is_file():
        write_json_atomically(first_header_path, headers[0])
    _write_records(headers_path, headers)


def _read_native_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    contents = path.read_bytes()
    if contents.startswith(b"\x1f\x8b"):
        records = read_jsonl(path, recover_partial=True)
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        return records
    lines = contents.splitlines(keepends=True)
    valid_lines: list[bytes] = []
    records: list[dict[str, object]] = []
    for line_index, raw_line in enumerate(lines):
        try:
            value: object = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if line_index != len(lines) - 1 or raw_line.endswith(b"\n"):
                raise ValueError(
                    f"{path}:{line_index + 1} is not a complete JSON record"
                ) from error
            torn_path = path.with_name(f"{path.name}.torn-{time.time_ns()}")
            torn_path.write_bytes(raw_line)
            path.write_bytes(b"".join(valid_lines))
            logger.warning("archived incomplete native ladder record at %s", torn_path)
            break
        if not isinstance(value, Mapping):
            raise TypeError(f"{path}:{line_index + 1} must contain an object")
        records.append(cast("dict[str, object]", value))
        valid_lines.append(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
    if records and contents and not contents.endswith(b"\n") and len(valid_lines) == len(lines):
        path.write_bytes(b"".join(valid_lines))
    return records


def _native_task_id(record: Mapping[str, object]) -> tuple[str, int]:
    key_value = record.get("key")
    if not isinstance(key_value, Mapping):
        raise TypeError("native ladder episode requires an object key")
    key = cast("Mapping[str, object]", key_value)
    slug = key.get("scenario_id")
    arm = key.get("arm")
    sample_index = key.get("sample_idx")
    if (
        not isinstance(slug, str)
        or not isinstance(arm, str)
        or type(sample_index) is not int
        or sample_index < 0
    ):
        raise ValueError("native ladder episode key is incomplete")
    return f"ladder:{slug}:{arm}", sample_index


def _validate_native_records(
    records: Sequence[Mapping[str, object]], task_ids: Sequence[str]
) -> dict[str, dict[str, object]]:
    expected = set(task_ids)
    found: dict[str, dict[str, object]] = {}
    for record in records:
        if record.get("record") not in {"ladder_episode", "ladder_episode_failure"}:
            raise ValueError("native ladder records may contain only episode rows")
        task_id, sample_index = _native_task_id(record)
        if task_id not in expected or sample_index != 0:
            raise ValueError("native ladder record is outside the selected wave")
        if task_id in found and _native_record_complete(found[task_id]):
            raise ValueError(f"duplicate native ladder episode for {task_id}")
        found[task_id] = dict(record)
    return found


def _native_record_complete(record: Mapping[str, object]) -> bool:
    return record.get("record") == "ladder_episode" or (
        record.get("record") == "ladder_episode_failure"
        and record.get("reason") == "thinking_budget_overshoot"
    )


def _adapt_native_record(  # noqa: PLR0913 - episode identity and evidence stay explicit
    native_record: Mapping[str, object],
    *,
    task_id: str,
    sample_index: int,
    seed: int,
    trace_path: Path,
    run_provenance_sha256: str,
    hint: _ResolvedHint,
) -> dict[str, object]:
    record = dict(native_record)
    score: RewardScore = score_ladder_task(native_record)
    record.update(
        {
            "record_schema": RECORD_SCHEMA_VERSION,
            "prompt_format_version": PROMPT_FORMAT_VERSION,
            "task_id": task_id,
            "family": "ladder",
            "variant": "ladder",
            "task_kind": "ladder",
            "payment_register": "ladder",
            "partition": "train",
            "sample_index": sample_index,
            "seed": seed,
            "trace_path": str(trace_path.resolve()),
            "organism_reward": {
                "reward": score.reward,
                "outcome": score.outcome.value,
                "evidence": asdict(score.evidence),
            },
            "organism_reward_outcome": score.outcome.value,
            "run_provenance_sha256": run_provenance_sha256,
            "hint": {"family": hint.family, "strength": hint.strength, "sha": hint.sha},
        }
    )
    return record


def _write_records(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _append_native_record(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(dict(record), sort_keys=True) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def run_ladder_wave(  # noqa: PLR0913, PLR0915 - explicit stage setup, collection, and output
    *,
    endpoint: str,
    model_path: str | Path,
    served_model_id: str,
    output_dir: Path,
    max_model_len: int,
    model_revision: str | None = None,
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
    concurrency: int = LADDER_WAVE_CONCURRENCY,
    server_args: Mapping[str, object] | None = None,
    hints_path: Path | None = None,
    hint_family: str | None = None,
    hint_strength: HintStrength = "none",
) -> dict[str, object]:
    """Generate or resume the eight real/claimed episodes and write label-refresh input rows."""
    if not endpoint or not served_model_id or max_model_len < 1 or concurrency < 1:
        raise ValueError("endpoint, served model, max model length, and concurrency must be valid")
    require_gitignored_output_dir(output_dir)
    hint = _resolve_hint(hints_path, hint_family, hint_strength)
    source = LadderEpisodeSource(
        endpoint=endpoint,
        model_path=model_path,
        served_model_id=served_model_id,
        revision=model_revision,
        output_dir=output_dir / "source",
        max_model_len=max_model_len,
        max_turns=LADDER_WAVE_MAX_TURNS,
        max_new_tokens=LADDER_WAVE_MAX_NEW_TOKENS,
        temperature=LADDER_WAVE_TEMPERATURE,
        top_p=LADDER_WAVE_TOP_P,
        top_k=LADDER_WAVE_TOP_K,
        presence_penalty=LADDER_WAVE_PRESENCE_PENALTY,
        arms=LADDER_WAVE_ARMS,
        episode_concurrency=concurrency,
        hint_family=hint.family,
        hint_strength=hint.strength,
        hint_text=hint.text,
        hint_sha=hint.sha,
        scenarios_dir=scenarios_dir,
        chat_template_kwargs={"reasoning_effort": LADDER_WAVE_REASONING_EFFORT},
        server_args=server_args,
        ignore_operational_identity=True,
    )
    scenarios = _unique_scenarios(source)
    task_ids = ladder_wave_task_ids(scenarios)
    provenance = _wave_provenance(
        source,
        endpoint=endpoint,
        model_path=model_path,
        model_revision=model_revision,
        served_model_id=served_model_id,
        max_model_len=max_model_len,
        scenarios=scenarios,
        task_ids=task_ids,
        concurrency=concurrency,
        hint=hint,
        hints_path=hints_path,
    )
    provenance_sha256 = _prepare_output(output_dir, provenance)
    expected_header = {
        "record": "ladder_run_header",
        "schema_version": runner.LADDER_SCHEMA_VERSION,
        "provenance_digest": source.config.provenance_digest,
        "provenance": source.config.provenance_payload(),
    }
    _prepare_native_headers(output_dir, expected_header)

    native_records_path = output_dir / NATIVE_RECORDS_FILENAME
    native_records = _validate_native_records(_read_native_records(native_records_path), task_ids)
    pending_task_ids = [
        task_id
        for task_id in task_ids
        if task_id not in native_records or not _native_record_complete(native_records[task_id])
    ]
    seeds = {task_id: runner_seed(task_id) for task_id in task_ids}
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="ladder-wave") as executor:
        futures = {
            executor.submit(source, task_id, seed=seeds[task_id], step=0, replica=0): task_id
            for task_id in pending_task_ids
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                sample = future.result()
            except (httpx2.TimeoutException, httpx2.NetworkError) as error:
                _, scenario_id, arm = task_id.split(":")
                failure: dict[str, object] = {
                    "record": "ladder_episode_failure",
                    "status": "failed",
                    "reason": "transport_error",
                    "key": runner.EpisodeKey(
                        source.backend.model_id, scenario_id, arm, 0
                    ).to_json(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                _append_native_record(native_records_path, failure)
                native_records[task_id] = failure
                logger.exception("ladder episode %s failed", task_id)
                continue
            if sample.trace_path is None:
                raise FileNotFoundError(f"ladder source returned no trace path for {task_id}")
            record = dict(sample.record)
            actual_task_id, sample_index = _native_task_id(record)
            if actual_task_id != task_id or sample_index != 0:
                raise ValueError("ladder source returned an episode with a different identity")
            _append_native_record(native_records_path, record)
            native_records[task_id] = record

    adapted_records: list[dict[str, object]] = []
    episode_counts: Counter[str] = Counter()
    for task_id in task_ids:
        native_record = native_records[task_id]
        if not _native_record_complete(native_record):
            continue
        trace_path_value = native_record.get("trace_path")
        if not isinstance(trace_path_value, str):
            raise TypeError("native ladder record requires a trace_path")
        record = _adapt_native_record(
            native_record,
            task_id=task_id,
            sample_index=0,
            seed=seeds[task_id],
            trace_path=Path(trace_path_value),
            run_provenance_sha256=provenance_sha256,
            hint=hint,
        )
        adapted_records.append(record)
        episode_counts[str(cast("Mapping[str, object]", record["key"])["arm"])] += 1
    _write_records(output_dir / RECORDS_FILENAME, adapted_records)
    failed_task_ids = [
        task_id for task_id in task_ids if not _native_record_complete(native_records[task_id])
    ]
    summary: dict[str, object] = {
        "failed_episodes": len(failed_task_ids),
        "failed_task_ids": failed_task_ids,
        "episodes": len(adapted_records),
        "scenario_count": len(scenarios),
        "episode_counts_by_arm": dict(sorted(episode_counts.items())),
        "sampling": cast("Mapping[str, object]", provenance["resume_identity"])["sampling"],
        "resumed_episodes": len(task_ids) - len(pending_task_ids),
        "torn_traces_recovered": source.torn_trace_count,
    }
    write_json_atomically(output_dir / SUMMARY_FILENAME, summary)
    if failed_task_ids:
        raise LadderWaveError(
            f"{len(failed_task_ids)} ladder episode(s) failed; completed outputs persisted"
        )
    return summary


def runner_seed(task_id: str) -> int:
    """Derive a stable sample seed from the task identity."""
    return deterministic_episode_seed(task_id, 0, 0)


def _parse_server_args(raw: str) -> Mapping[str, object]:
    value: object = json.loads(raw)
    if not isinstance(value, Mapping):
        raise argparse.ArgumentTypeError("server args JSON must be an object")
    return cast("Mapping[str, object]", value)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True, help="local model path or cached repository id")
    parser.add_argument("--served-model-id", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios-dir", type=Path, default=DEFAULT_SCENARIOS_DIR)
    parser.add_argument("--episode-concurrency", type=int, default=LADDER_WAVE_CONCURRENCY)
    parser.add_argument("--server-args-json", type=_parse_server_args)
    parser.add_argument("--hints", type=Path)
    parser.add_argument("--hint-family")
    parser.add_argument("--hint-strength", choices=("vague", "specific", "none"), default="none")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ladder generation stage against an already managed vLLM endpoint."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        run_ladder_wave(
            endpoint=args.endpoint,
            model_path=args.model,
            served_model_id=args.served_model_id,
            model_revision=args.model_revision,
            output_dir=args.output_dir,
            max_model_len=args.max_model_len,
            scenarios_dir=args.scenarios_dir,
            concurrency=args.episode_concurrency,
            server_args=args.server_args_json,
            hints_path=args.hints,
            hint_family=args.hint_family,
            hint_strength=args.hint_strength,
        )
    except LadderWaveError:
        logger.exception("ladder wave failed after writing completed outputs")
        return 1
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
