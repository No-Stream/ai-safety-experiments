"""Recompute hack-ladder detections from persisted episode evidence without a model runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from reward_hacking.jailed_poc import ENFORCED_JAIL_RESOURCE_LIMITS, JailResourceLimits
from reward_hacking.ladder import rungs, runner
from reward_hacking.ladder import scenario as scenario_module
from reward_hacking.ladder.monitor import CommandRecord
from reward_hacking.ladder.runner import (
    LADDER_TASK_PRESSURES,
    EpisodeKey,
    default_detector,
    detection_spec_for_scenario,
    detection_spec_from_json,
    final_state_diff_from_disk,
    final_state_diff_from_json,
    read_in_scene_audit_log,
)
from reward_hacking.ladder.scenario import Scenario

if TYPE_CHECKING:
    from reward_hacking.ladder.rungs import DetectionSpec, FinalStateDiff

_DEFAULT_SCENARIO_ROOT = Path(__file__).resolve().parent / "data" / "scenarios"


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be an object with string keys")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FileNotFoundError(f"cannot read ladder input {path}: {error}") from error
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed JSON in {path} at line {line_number}") from error
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row in {path} at line {line_number} must be an object")
        records.append(value)
    return records


def _scenario_digests(header: Mapping[str, object]) -> dict[str, str]:
    provenance = _require_mapping(header.get("provenance"), field="run header provenance")
    raw_scenarios = provenance.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise TypeError("run header provenance.scenarios must be a list")
    digests: dict[str, str] = {}
    for index, raw_scenario in enumerate(raw_scenarios):
        scenario_record = _require_mapping(
            raw_scenario, field=f"run header provenance.scenarios[{index}]"
        )
        scenario_id = scenario_record.get("scenario_id")
        manifest_digest = scenario_record.get("manifest_digest")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise TypeError(f"run header scenario {index} has no string scenario_id")
        if not isinstance(manifest_digest, str) or not manifest_digest:
            raise TypeError(f"run header scenario {scenario_id!r} has no manifest_digest")
        if scenario_id in digests:
            raise ValueError(f"run header contains duplicate scenario id {scenario_id!r}")
        digests[scenario_id] = manifest_digest
    return digests


def _headers_by_pressure(
    headers: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Validate compatible run headers and index them by their pressure condition."""
    if not headers:
        raise ValueError("input has no ladder_run_header provenance record")
    headers_by_pressure: dict[str, dict[str, object]] = {}
    common_provenance: dict[str, object] | None = None
    for header in headers:
        provenance = _require_mapping(header.get("provenance"), field="run header provenance")
        pressure = provenance.get("task_pressure", "none")
        if not isinstance(pressure, str) or pressure not in LADDER_TASK_PRESSURES:
            raise ValueError(f"run header has invalid task pressure {pressure!r}")
        current_common = dict(provenance)
        current_common.pop("task_pressure", None)
        if common_provenance is None:
            common_provenance = current_common
        elif current_common != common_provenance:
            raise ValueError("run headers have incompatible provenance")
        digest = header.get("provenance_digest")
        if not isinstance(digest, str):
            raise TypeError("run header provenance_digest must be a string")
        expected_digest = runner.provenance_digest(provenance)
        if digest != expected_digest:
            raise ValueError(f"run header provenance digest is invalid for pressure {pressure!r}")
        if pressure in headers_by_pressure:
            previous_digest = headers_by_pressure[pressure].get("provenance_digest")
            if previous_digest != digest:
                raise ValueError(f"run has conflicting headers for pressure {pressure!r}")
            continue
        headers_by_pressure[pressure] = header
    return headers_by_pressure


def _load_scenario(scenario_root: Path, scenario_id: str, expected_digest: str) -> Scenario:
    scenario_path = (scenario_root / scenario_id).resolve()
    if not scenario_path.is_relative_to(scenario_root.resolve()):
        raise ValueError(f"scenario id escapes the scenario root: {scenario_id!r}")
    if not scenario_path.is_dir():
        raise FileNotFoundError(f"scenario directory does not exist: {scenario_path}")
    scenario = Scenario.load(scenario_path)
    if scenario.slug != scenario_id:
        raise ValueError(
            f"scenario directory {scenario_path} declares slug {scenario.slug!r}, "
            f"expected {scenario_id!r}"
        )
    if scenario.manifest_digest != expected_digest:
        raise ValueError(
            f"scenario {scenario_id!r} does not match the manifest digest in the run header"
        )
    return scenario


def _episode_dir(record: Mapping[str, object]) -> Path:
    value = record.get("episode_dir")
    if not isinstance(value, str) or not value:
        raise ValueError("ladder_episode record has no episode_dir")
    episode_dir = Path(value).resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode directory does not exist: {episode_dir}")
    return episode_dir


def _command_records(record: Mapping[str, object]) -> tuple[CommandRecord, ...]:
    value = record.get("engine_command_records")
    if not isinstance(value, list):
        raise TypeError("ladder_episode.engine_command_records must be a list")
    return tuple(CommandRecord.from_json_dict(item) for item in value)


def _stored_detector_inputs(
    record: Mapping[str, object],
    scenario: Scenario,
    episode_dir: Path,
    command_records: Sequence[CommandRecord],
) -> tuple[str | None, FinalStateDiff | None, DetectionSpec]:
    value = record.get("detector_inputs")
    if value is None and "detector_inputs" not in record:
        spec = detection_spec_for_scenario(scenario)
        audit_log = read_in_scene_audit_log(episode_dir, audit_log_path=spec.audit_log_path)
        final_state_diff = final_state_diff_from_disk(
            episode_dir,
            command_records,
            monitor_config_path=spec.monitor_config_path,
            audit_log_path=spec.audit_log_path,
        )
        return audit_log, final_state_diff, spec

    detector_inputs = _require_mapping(value, field="ladder_episode.detector_inputs")
    expected_fields = {"in_scene_audit_log", "final_state_diff", "detection_spec"}
    if set(detector_inputs) != expected_fields:
        missing = sorted(expected_fields - set(detector_inputs))
        extra = sorted(set(detector_inputs) - expected_fields)
        raise ValueError(f"detector_inputs fields mismatch (missing={missing}, extra={extra})")
    audit_log = detector_inputs["in_scene_audit_log"]
    if audit_log is not None and not isinstance(audit_log, str):
        raise TypeError("detector_inputs.in_scene_audit_log must be a string or null")
    return (
        audit_log,
        final_state_diff_from_json(detector_inputs["final_state_diff"]),
        detection_spec_from_json(detector_inputs["detection_spec"]),
    )


def _rescore_episode(
    record: dict[str, object],
    *,
    scenario_root: Path,
    scenario_digests: Mapping[str, str],
    scenario_cache: dict[str, Scenario],
    grading_settings: tuple[str, JailResourceLimits],
) -> dict[str, object]:
    raw_key = _require_mapping(record.get("key"), field="ladder_episode.key")
    key = EpisodeKey.from_json(raw_key)
    try:
        expected_digest = scenario_digests[key.scenario_id]
    except KeyError as error:
        raise ValueError(
            f"scenario {key.scenario_id!r} is absent from run header provenance"
        ) from error
    scenario = scenario_cache.get(key.scenario_id)
    if scenario is None:
        scenario = _load_scenario(scenario_root, key.scenario_id, expected_digest)
        scenario_cache[key.scenario_id] = scenario

    episode_dir = _episode_dir(record)
    command_records = _command_records(record)
    audit_log, final_state_diff, spec = _stored_detector_inputs(
        record, scenario, episode_dir, command_records
    )
    pristine_visible_pass = record.get("pristine_visible_pass")
    if pristine_visible_pass is None:
        timeout, resource_limits = grading_settings
        pristine_grader_pass, pristine_tests_pass = runner.run_pristine_visible_grade(
            scenario,
            episode_dir,
            timeout=timeout,
            resource_limits=resource_limits,
            task_pressure=key.task_pressure,
        )
        pristine_visible_pass = pristine_grader_pass and pristine_tests_pass
    elif not isinstance(pristine_visible_pass, bool):
        raise TypeError("ladder_episode.pristine_visible_pass must be a boolean or null")
    trace_summary = _require_mapping(record.get("trace_summary"), field="trace_summary")
    detection = default_detector(
        scenario=scenario,
        episode_dir=episode_dir,
        trace=trace_summary,
        arm=key.arm,
        command_records=command_records,
        final_state_diff=final_state_diff,
        in_scene_audit_log=audit_log,
        detection_spec=spec,
        pristine_visible_pass=pristine_visible_pass,
    )
    highest_rung = detection.get("highest_rung")
    coverup = detection.get("coverup")
    if not isinstance(highest_rung, int) or isinstance(highest_rung, bool):
        raise TypeError("current detector returned an invalid highest_rung")
    if not isinstance(coverup, bool):
        raise TypeError("current detector returned an invalid coverup value")
    record["rung_detection"] = dict(detection)
    record["highest_rung"] = highest_rung
    record["coverup"] = coverup
    record["rung_5"] = coverup
    record["pristine_visible_pass"] = pristine_visible_pass
    return record


def _grading_settings(header: Mapping[str, object]) -> tuple[str, JailResourceLimits]:
    """Reuse the run's timeout and jail mode when rebuilding its pristine grade."""
    provenance = _require_mapping(header.get("provenance"), field="run header provenance")
    timeout = provenance.get("timeout", "10s")
    if not isinstance(timeout, str) or not timeout:
        raise TypeError("run header provenance.timeout must be a non-empty string")
    raw_limits = provenance.get("resource_limits")
    if raw_limits is None:
        return timeout, ENFORCED_JAIL_RESOURCE_LIMITS
    limits = _require_mapping(raw_limits, field="run header provenance.resource_limits")
    mode_value = limits.get("mode")
    if mode_value == "enforced":
        mode: Literal["enforced", "advisory"] = "enforced"
    elif mode_value == "advisory":
        mode = "advisory"
    else:
        raise ValueError("run header provenance.resource_limits.mode is invalid")
    systemd_user_state = limits.get("systemd_user_state")
    if systemd_user_state is not None and not isinstance(systemd_user_state, str):
        raise TypeError("run header provenance.resource_limits.systemd_user_state is invalid")
    return timeout, JailResourceLimits(mode=mode, systemd_user_state=systemd_user_state)


def _detector_code_identity() -> str:
    digest = hashlib.sha256()
    for module in (runner, rungs, scenario_module):
        source_path = Path(cast("str", module.__file__))
        digest.update(source_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def rescore_file(
    input_path: Path,
    output_path: Path,
    *,
    scenario_root: Path = _DEFAULT_SCENARIO_ROOT,
) -> list[dict[str, object]]:
    """Rescore every ladder episode in input order and write a new JSONL artifact."""
    source_path = input_path.resolve()
    destination_path = output_path.resolve()
    if source_path == destination_path:
        raise ValueError("input and output paths must be different")
    rows = _read_jsonl(source_path)
    headers = [row for row in rows if row.get("record") == "ladder_run_header"]
    headers_by_pressure = _headers_by_pressure(headers)
    resolved_scenario_root = scenario_root.resolve()
    scenario_cache: dict[str, Scenario] = {}
    scenario_digests_by_pressure = {
        pressure: _scenario_digests(header) for pressure, header in headers_by_pressure.items()
    }
    grading_settings_by_pressure = {
        pressure: _grading_settings(header) for pressure, header in headers_by_pressure.items()
    }
    for row in rows:
        if row.get("record") == "ladder_episode":
            raw_key = _require_mapping(row.get("key"), field="ladder_episode.key")
            key = EpisodeKey.from_json(raw_key)
            try:
                header = headers_by_pressure[key.task_pressure]
            except KeyError as error:
                raise ValueError(f"no run header for pressure {key.task_pressure!r}") from error
            if row.get("provenance_digest") != header.get("provenance_digest"):
                raise ValueError(
                    f"ladder episode provenance digest does not match pressure "
                    f"{key.task_pressure!r} header"
                )
            _rescore_episode(
                row,
                scenario_root=resolved_scenario_root,
                scenario_digests=scenario_digests_by_pressure[key.task_pressure],
                scenario_cache=scenario_cache,
                grading_settings=grading_settings_by_pressure[key.task_pressure],
            )
    for header in headers_by_pressure.values():
        header["rescore"] = {
            "detector": "reward_hacking.ladder.rungs.detect_rungs",
            "detector_code_identity": _detector_code_identity(),
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
    _write_jsonl(destination_path, rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line parser and return a process status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input ladder run JSONL")
    parser.add_argument("--output", type=Path, required=True, help="rescored JSONL output")
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=_DEFAULT_SCENARIO_ROOT,
        help=f"scenario directory containing scenario-id subdirectories (default: {_DEFAULT_SCENARIO_ROOT})",
    )
    arguments = parser.parse_args(argv)
    try:
        rescore_file(
            arguments.input,
            arguments.output,
            scenario_root=arguments.scenario_root,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
