"""Create and validate immutable, labels-only episode sidecars."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from reward_hacking.distill.call_history import (
    CandidateContract,
    analyze_call_history,
    recover_candidate_contract,
)
from reward_hacking.distill.hack_strategy import (
    TAXONOMY_VERSION,
    StrengthenedTruthEvidence,
    _reconstruct_final_source,  # pyright: ignore[reportPrivateUsage]
    _recorded_true_pass,  # pyright: ignore[reportPrivateUsage]
    label_persisted_episode,
    read_strengthened_truth_sidecar,
)
from reward_hacking.distill.table_provenance import confirmed_visible_table
from reward_hacking.harness.evidence_store import read_jsonl
from reward_hacking.organism_rl.strengthened_truth_grade import recorded_artifact_sha256

LABEL_REFRESH_FORMAT_VERSION = "episode_label_refresh_v1"
LABELS_FILENAME = "labels.jsonl"
MANIFEST_FILENAME = "manifest.json"
logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LabelRefreshResult:
    """Paths for the immutable derived labels and their manifest."""

    labels_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class _StrengthenedTruthBinding:
    task_id: str
    artifact_sha256: str | None
    true_pass: bool | None
    evidence: StrengthenedTruthEvidence
    source_sha256: str


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_records(records_path: Path) -> tuple[bytes, list[tuple[bytes, dict[str, object]]]]:
    raw_records = records_path.read_bytes()
    if not raw_records:
        raise ValueError(f"{records_path} contains no episode records")
    stored_records = (
        gzip.decompress(raw_records) if raw_records.startswith(b"\x1f\x8b") else raw_records
    )
    hydrated_records = iter(read_jsonl(records_path))
    parsed_records: list[tuple[bytes, dict[str, object]]] = []
    for line_number, raw_line in enumerate(stored_records.splitlines(keepends=True), start=1):
        if not raw_line.strip():
            raise ValueError(f"{records_path}:{line_number} is an empty records line")
        parsed_records.append((raw_line, next(hydrated_records)))
    return raw_records, parsed_records


def _episode_identity(
    record: Mapping[str, object], context: str
) -> tuple[str, int, dict[str, object]]:
    episode_key = record.get("key")
    if not isinstance(episode_key, Mapping) or not episode_key:
        raise TypeError(f"{context} requires an object episode key")
    sample_index = record.get("sample_index")
    if type(sample_index) is not int or sample_index < 0:
        raise TypeError(f"{context} requires a nonnegative sample_index")
    key_sample_index = episode_key.get("sample_index")
    if key_sample_index is not None and key_sample_index != sample_index:
        raise ValueError(f"{context} episode key and sample_index disagree")
    key_json = json.dumps(
        dict(episode_key), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return key_json, sample_index, dict(episode_key)


def derived_label_join_key(record: Mapping[str, object]) -> tuple[str, int]:
    """Return the validated sidecar join key for one persisted episode record."""
    key_json, sample_index, _episode_key = _episode_identity(record, "episode record")
    return key_json, sample_index


def _trace_path(records_path: Path, record: Mapping[str, object], context: str) -> Path:
    trace_path_value = record.get("trace_path")
    if not isinstance(trace_path_value, str) or not trace_path_value:
        raise TypeError(f"{context} requires a trace_path")
    trace_path = Path(trace_path_value)
    if not trace_path.is_absolute():
        trace_path = records_path.parent / trace_path
    if not trace_path.is_file():
        raise FileNotFoundError(f"{context} trace does not exist: {trace_path}")
    return trace_path


def _evidence_digests(
    records_path: Path,
    raw_line: bytes,
    record: Mapping[str, object],
    row_index: int,
) -> dict[str, str | None]:
    context = f"{records_path}:row {row_index}"
    trace_path = _trace_path(records_path, record, context)
    summary = record.get("trace_summary")
    if not isinstance(summary, Mapping):
        raise TypeError(f"{context} requires a trace_summary object")
    artifact_digest = recorded_artifact_sha256(records_path.parent, record, summary)
    return {
        "source_record_sha256": _sha256(raw_line),
        "trace_sha256": _sha256(trace_path.read_bytes()),
        "artifact_sha256": artifact_digest,
    }


def _read_trace_records(trace_path: Path) -> list[dict[str, object]]:
    payload = trace_path.read_bytes()
    stored = gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload
    for line_number, line in enumerate(stored.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{trace_path}:{line_number} is an empty trace line")
    return read_jsonl(trace_path)


def _source_from_trace(trace_records: Sequence[Mapping[str, object]]) -> str | None:
    return _reconstruct_final_source({}, trace_records)


def _candidate_contract_audit(contract: CandidateContract) -> dict[str, object]:
    input_case_digests = [
        _sha256(
            json.dumps(
                input_case,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        )
        for input_case in contract.input_cases
    ]
    return {
        "entry_point": contract.entry_point,
        "deterministic": contract.deterministic,
        "limitations": contract.limitations,
        "task_definition_sha256": contract.task_definition_sha256,
        "input_case_count": len(contract.input_cases),
        "input_case_digests": input_case_digests,
    }


def _episode_analysis_audit(
    record: Mapping[str, object], trace_records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    contract = recover_candidate_contract(record, trace_records)
    source = _reconstruct_final_source(record, trace_records)
    table_provenance = confirmed_visible_table(
        source,
        trace_records,
        contract.entry_point,
        reconstruct_source=_source_from_trace,
    )
    if source is None or contract.entry_point is None:
        limitations = list(contract.limitations)
        if source is None:
            limitations.append("digest-verified final source could not be reconstructed")
        if contract.entry_point is None:
            limitations.append(
                "candidate entry point could not be recovered from persisted evidence"
            )
        analysis: dict[str, object] = {
            "state": "insufficient",
            "evidence": [],
            "limitations": list(dict.fromkeys(limitations)),
        }
    else:
        analysis = asdict(
            analyze_call_history(
                source,
                entry_point=contract.entry_point,
                deterministic_contract=contract.deterministic,
                input_cases=contract.input_cases,
            )
        )
    return {
        "candidate_contract": _candidate_contract_audit(contract),
        "call_history": analysis,
        "visible_table": asdict(table_provenance),
    }


def _code_revision() -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("git executable is required to record the labeler code revision")
    result = subprocess.run(  # noqa: S603 - argv is fixed and git is resolved from PATH.
        [git_executable, "rev-parse", "HEAD"],
        cwd=_repository_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if not revision:
        raise ValueError("git did not return a code revision")
    return revision


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_private_output_dir(output_dir: Path) -> None:
    repository_root = _repository_root()
    try:
        repository_relative_output = output_dir.relative_to(repository_root)
    except ValueError:
        return
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError(
            "git executable is required to validate the derived-label output path"
        )
    result = subprocess.run(  # noqa: S603 - argv is fixed and git is resolved from PATH.
        [
            git_executable,
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            str(repository_relative_output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 1:
        raise ValueError("derived-label output inside the repository must be gitignored")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )


def _validate_unique_episode_identities(
    records_path: Path, parsed_records: Sequence[tuple[bytes, dict[str, object]]]
) -> None:
    seen: set[tuple[str, int]] = set()
    for row_index, (_raw_line, record) in enumerate(parsed_records, start=1):
        identity = _episode_identity(record, f"{records_path}:row {row_index}")
        identity_key = (identity[0], identity[1])
        if identity_key in seen:
            raise ValueError(f"{records_path}: duplicate episode identity at row {row_index}")
        seen.add(identity_key)


def _read_strengthened_truth_bindings(
    sidecar_path: Path,
) -> tuple[bytes, dict[str, _StrengthenedTruthBinding]]:
    sidecar_bytes = sidecar_path.read_bytes()
    evidence_by_episode_id = read_strengthened_truth_sidecar(sidecar_path)
    bindings: dict[str, _StrengthenedTruthBinding] = {}
    for row_index, raw_line in enumerate(sidecar_bytes.splitlines(keepends=True), start=1):
        row_value: object = json.loads(raw_line)
        if not isinstance(row_value, dict):
            raise TypeError(f"{sidecar_path}:{row_index} must contain a JSON object")
        row = cast("dict[str, object]", row_value)
        episode_id = row.get("episode_id")
        task_id = row.get("task_id")
        artifact_sha256 = row.get("artifact_sha256")
        true_pass = row.get("true_pass")
        if not isinstance(episode_id, str) or not episode_id:
            raise TypeError(f"{sidecar_path}:{row_index} requires an episode_id")
        if not isinstance(task_id, str) or not task_id:
            raise TypeError(f"{sidecar_path}:{row_index} requires a task_id")
        if artifact_sha256 is not None and not isinstance(artifact_sha256, str):
            raise TypeError(f"{sidecar_path}:{row_index} artifact_sha256 must be a string or null")
        if true_pass is not None and type(true_pass) is not bool:
            raise TypeError(f"{sidecar_path}:{row_index} true_pass must be a boolean or null")
        evidence = evidence_by_episode_id[episode_id]
        bindings[episode_id] = _StrengthenedTruthBinding(
            task_id=task_id,
            artifact_sha256=artifact_sha256,
            true_pass=true_pass,
            evidence=evidence,
            source_sha256=_sha256(raw_line),
        )
    if set(bindings) != set(evidence_by_episode_id):
        raise ValueError(f"{sidecar_path} has inconsistent episode identities")
    return sidecar_bytes, bindings


def _strengthened_truth_episode_id(record: Mapping[str, object], context: str) -> str | None:
    if record.get("variant") not in {"original", "golden", "timed"} or record.get("family") not in {
        "aisi",
        "ilcb",
    }:
        return None
    summary = record.get("trace_summary", record.get("episode_record"))
    if not isinstance(summary, Mapping):
        raise TypeError(f"{context} requires a trace summary for strengthened truth")
    episode_id = summary.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise TypeError(f"{context} requires an episode_id for strengthened truth")
    return episode_id


def _strengthened_truth_for_record(
    record: Mapping[str, object],
    *,
    context: str,
    artifact_sha256: str | None,
    bindings: Mapping[str, _StrengthenedTruthBinding],
) -> tuple[StrengthenedTruthEvidence | None, str | None]:
    episode_id = _strengthened_truth_episode_id(record, context)
    if episode_id is None:
        return None, None
    binding = bindings.get(episode_id)
    if binding is None:
        raise ValueError(f"{context} is missing strengthened-truth sidecar evidence")
    task_id = record.get("task_id")
    if binding.task_id != task_id:
        raise ValueError(f"{context} strengthened-truth task id does not match")
    if binding.artifact_sha256 != artifact_sha256:
        raise ValueError(f"{context} strengthened-truth artifact digest does not match")
    if binding.true_pass != _recorded_true_pass(record):
        raise ValueError(f"{context} strengthened-truth original verdict does not match")
    return binding.evidence, binding.source_sha256


def _load_strengthened_truth_inputs(
    strengthened_truth_path: Path | None, output_dir: Path
) -> tuple[Path | None, bytes | None, dict[str, _StrengthenedTruthBinding] | None]:
    if strengthened_truth_path is None:
        return None, None, None
    strengthened_truth_path = strengthened_truth_path.resolve(strict=True)
    if strengthened_truth_path.is_relative_to(output_dir):
        raise ValueError("strengthened-truth sidecar and label output directory overlap")
    strengthened_truth_bytes, bindings = _read_strengthened_truth_bindings(strengthened_truth_path)
    return strengthened_truth_path, strengthened_truth_bytes, bindings


def _truth_for_label(
    record: Mapping[str, object],
    *,
    context: str,
    artifact_sha256: str | None,
    bindings: dict[str, _StrengthenedTruthBinding] | None,
    used_episode_ids: set[str],
) -> tuple[StrengthenedTruthEvidence | None, str | None]:
    if bindings is None:
        return None, None
    strengthened_truth, source_sha256 = _strengthened_truth_for_record(
        record,
        context=context,
        artifact_sha256=artifact_sha256,
        bindings=bindings,
    )
    if strengthened_truth is not None:
        summary = record.get("trace_summary", record.get("episode_record"))
        episode_id = cast("str", cast("Mapping[str, object]", summary)["episode_id"])
        if episode_id in used_episode_ids:
            raise ValueError("multiple records match one strengthened-truth episode")
        used_episode_ids.add(episode_id)
    return strengthened_truth, source_sha256


def _validate_strengthened_truth_join_complete(
    bindings: dict[str, _StrengthenedTruthBinding] | None,
    used_episode_ids: set[str],
) -> None:
    if bindings is None:
        return
    unused_episode_ids = set(bindings).difference(used_episode_ids)
    if unused_episode_ids:
        raise ValueError(
            "strengthened-truth sidecar has rows without matching original records: "
            f"{sorted(unused_episode_ids)}"
        )


def _validate_episode_inputs_unchanged(
    records_path: Path,
    raw_records: bytes,
    parsed_records: list[tuple[bytes, dict[str, object]]],
    before_digests: list[dict[str, str | None]],
) -> None:
    current_records = records_path.read_bytes()
    after_digests = [
        _evidence_digests(records_path, raw_line, record, row_index)
        for row_index, (raw_line, record) in enumerate(parsed_records, start=1)
    ]
    if current_records != raw_records or after_digests != before_digests:
        raise ValueError(
            "source records, traces or artifacts changed during the labels-only refresh"
        )


def _validate_strengthened_truth_unchanged(
    strengthened_truth_path: Path | None, strengthened_truth_bytes: bytes | None
) -> None:
    if strengthened_truth_path is not None and strengthened_truth_path.read_bytes() != (
        strengthened_truth_bytes
    ):
        raise ValueError("strengthened-truth sidecar changed during the labels-only refresh")


def refresh_labels(
    records_path: Path,
    output_dir: Path,
    *,
    strengthened_truth_path: Path | None = None,
) -> LabelRefreshResult:
    """Write fresh derived labels without changing source records, traces or artifacts."""
    records_path = records_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"derived-label output directory already exists: {output_dir}")
    if output_dir.is_relative_to(records_path.parent) or records_path.is_relative_to(output_dir):
        raise ValueError(
            "derived-label output directory cannot overlap the original episode directory"
        )
    _validate_private_output_dir(output_dir)

    (
        strengthened_truth_path,
        strengthened_truth_bytes,
        strengthened_truth_bindings,
    ) = _load_strengthened_truth_inputs(strengthened_truth_path, output_dir)
    used_strengthened_truth: set[str] = set()

    raw_records, parsed_records = _read_records(records_path)
    _validate_unique_episode_identities(records_path, parsed_records)
    source_records_digest = _sha256(raw_records)
    before_digests = [
        _evidence_digests(records_path, raw_line, record, row_index)
        for row_index, (raw_line, record) in enumerate(parsed_records, start=1)
    ]
    labels: list[dict[str, object]] = []
    for row_index, (raw_line, record) in enumerate(parsed_records, start=1):
        _key_json, sample_index, episode_key = _episode_identity(
            record, f"{records_path}:row {row_index}"
        )
        trace_path = _trace_path(records_path, record, f"{records_path}:row {row_index}")
        trace_records = _read_trace_records(trace_path)
        normalized_record = {**record, "trace_path": str(trace_path.resolve())}
        strengthened_truth, strengthened_truth_source_sha256 = _truth_for_label(
            record,
            context=f"{records_path}:row {row_index}",
            artifact_sha256=before_digests[row_index - 1]["artifact_sha256"],
            bindings=strengthened_truth_bindings,
            used_episode_ids=used_strengthened_truth,
        )
        assessment = label_persisted_episode(
            normalized_record,
            records_path=records_path,
            strengthened_truth=strengthened_truth,
        )
        labels.append(
            {
                "episode_key": episode_key,
                "sample_index": sample_index,
                "source_record_sha256": _sha256(raw_line),
                "trace_sha256": before_digests[row_index - 1]["trace_sha256"],
                "artifact_sha256": before_digests[row_index - 1]["artifact_sha256"],
                "strengthened_truth_source_sha256": strengthened_truth_source_sha256,
                "labeler_version": TAXONOMY_VERSION,
                "hack_strategy": assessment.to_json_dict(),
                "analysis": _episode_analysis_audit(normalized_record, trace_records),
            }
        )

    _validate_strengthened_truth_join_complete(strengthened_truth_bindings, used_strengthened_truth)
    _validate_episode_inputs_unchanged(records_path, raw_records, parsed_records, before_digests)
    _validate_strengthened_truth_unchanged(
        strengthened_truth_path,
        strengthened_truth_bytes,
    )

    labels_bytes = "".join(json.dumps(label, sort_keys=True) + "\n" for label in labels).encode()
    manifest = {
        "format_version": LABEL_REFRESH_FORMAT_VERSION,
        "source_records_sha256": source_records_digest,
        "strengthened_truth_source_sha256": (
            _sha256(strengthened_truth_bytes) if strengthened_truth_bytes is not None else None
        ),
        "labeler_version": TAXONOMY_VERSION,
        "code_revision": _code_revision(),
        "row_count": len(labels),
        "labels_sha256": _sha256(labels_bytes),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    labels_path = output_dir / LABELS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    labels_path.write_bytes(labels_bytes)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return LabelRefreshResult(labels_path=labels_path, manifest_path=manifest_path)


def _read_manifest(labels_path: Path) -> dict[str, object]:
    manifest_path = labels_path.with_name(MANIFEST_FILENAME)
    manifest_value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise TypeError(f"{manifest_path} must contain a JSON object")
    manifest = cast("dict[str, object]", manifest_value)
    if manifest.get("format_version") != LABEL_REFRESH_FORMAT_VERSION:
        raise ValueError(f"{manifest_path} has an unsupported label-sidecar format")
    if manifest.get("labeler_version") != TAXONOMY_VERSION:
        raise ValueError(f"{manifest_path} labeler version does not match {TAXONOMY_VERSION}")
    if not isinstance(manifest.get("code_revision"), str) or not manifest["code_revision"]:
        raise ValueError(f"{manifest_path} is missing the labeler code revision")
    row_count = manifest.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise ValueError(f"{manifest_path} has an invalid row count")
    return manifest


def load_derived_labels(
    records_path: Path, labels_path: Path
) -> dict[tuple[str, int], dict[str, object]]:
    """Validate and return sidecar labels keyed by the persisted episode key and sample index."""
    raw_records, parsed_records = _read_records(records_path)
    _validate_unique_episode_identities(records_path, parsed_records)
    manifest = _read_manifest(labels_path)
    if manifest.get("source_records_sha256") != _sha256(raw_records):
        raise ValueError(f"{records_path} digest does not match the label sidecar manifest")
    labels_bytes = labels_path.read_bytes()
    if manifest.get("labels_sha256") != _sha256(labels_bytes):
        raise ValueError(f"{labels_path} digest does not match its manifest")
    label_lines = labels_bytes.splitlines(keepends=True)
    if len(label_lines) != manifest["row_count"]:
        raise ValueError(f"{labels_path} row count does not match its manifest")
    if len(label_lines) != len(parsed_records):
        raise ValueError(f"{labels_path} has missing or extra episode joins")

    derived_labels: dict[tuple[str, int], dict[str, object]] = {}
    for row_index, (raw_line, source_record) in enumerate(parsed_records, start=1):
        label = _parse_label_line(labels_path, row_index, label_lines[row_index - 1])
        join_key = _validate_label_identity(labels_path, row_index, source_record, label)
        if join_key in derived_labels:
            raise ValueError(f"{labels_path} contains a duplicate episode join")
        _validate_label_evidence(records_path, row_index, raw_line, source_record, label)
        _validate_label_payload(labels_path, row_index, label)
        derived_labels[join_key] = label

    if len(derived_labels) != len(parsed_records):
        raise ValueError(f"{labels_path} has missing or duplicate episode joins")
    return derived_labels


def _parse_label_line(labels_path: Path, row_index: int, label_line: bytes) -> dict[str, object]:
    if not label_line.strip():
        raise ValueError(f"{labels_path}:{row_index} is an empty label line")
    label_value: object = json.loads(label_line)
    if not isinstance(label_value, dict):
        raise TypeError(f"{labels_path}:{row_index} must contain a JSON object")
    return cast("dict[str, object]", label_value)


def _validate_label_identity(
    labels_path: Path,
    row_index: int,
    source_record: Mapping[str, object],
    label: Mapping[str, object],
) -> tuple[str, int]:
    key_json, sample_index, episode_key = _episode_identity(
        source_record, f"{labels_path}:row {row_index}"
    )
    if label.get("episode_key") != episode_key or label.get("sample_index") != sample_index:
        raise ValueError(
            f"{labels_path}:{row_index} episode identity does not match its source row"
        )
    return key_json, sample_index


def _validate_label_evidence(
    records_path: Path,
    row_index: int,
    raw_line: bytes,
    source_record: Mapping[str, object],
    label: Mapping[str, object],
) -> None:
    if label.get("labeler_version") != TAXONOMY_VERSION:
        raise ValueError(f"{records_path}:row {row_index} labeler version is stale")
    digests = _evidence_digests(records_path, raw_line, source_record, row_index)
    if any(label.get(name) != digest for name, digest in digests.items()):
        raise ValueError(
            f"{records_path}:row {row_index} source, trace or artifact digest is stale"
        )


def _validate_label_payload(labels_path: Path, row_index: int, label: Mapping[str, object]) -> None:
    if not isinstance(label.get("hack_strategy"), Mapping):
        raise TypeError(f"{labels_path}:{row_index} is missing a hack_strategy assessment")
    analysis = label.get("analysis")
    if not isinstance(analysis, Mapping):
        raise TypeError(f"{labels_path}:{row_index} is missing private analysis")
    for key in ("candidate_contract", "call_history", "visible_table"):
        if not isinstance(analysis.get(key), Mapping):
            raise TypeError(f"{labels_path}:{row_index} is missing private {key} analysis")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strengthened-truth", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Refresh labels to a new sidecar directory and log its paths."""
    args = _argument_parser().parse_args(argv)
    result = refresh_labels(
        args.records,
        args.output_dir,
        strengthened_truth_path=args.strengthened_truth,
    )
    logging.basicConfig(level=logging.INFO)
    logger.info("labels=%s manifest=%s", result.labels_path, result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
