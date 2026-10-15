"""Trusted SpreadsheetBench worker for isolated workbook evaluation."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import posixpath
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from xml.etree.ElementTree import ParseError

import defusedxml.ElementTree
from defusedxml.common import DefusedXmlException

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

logger: logging.Logger = logging.getLogger(__name__)

_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DOCUMENT_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_UPSTREAM_LIMITATION = "upstream_value_comparison_does_not_establish_formula_or_format_requirements"


@dataclass(frozen=True)
class _AssessmentRequest:
    evaluation_root: Path
    reference_bytes: bytes
    submission_bytes: bytes
    full_cells: tuple[tuple[str, str], ...]
    proxy_outputs: tuple[tuple[str, str], ...]
    proxy_cells: tuple[str, ...]
    instruction_type: str
    answer_position: str
    compare_workbooks: Callable[..., Any]
    operational_only: bool = False


class EvaluatorError(ValueError):
    """A known failure that makes the assessment unmeasured."""

    def __init__(self, kind: str) -> None:
        """Store a stable machine-readable failure kind."""
        super().__init__(kind)
        self.kind = kind


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _required_bytes(workdir: Path, name: str) -> bytes:
    try:
        return (workdir / name).read_bytes()
    except FileNotFoundError as error:
        raise EvaluatorError(f"{name.removesuffix('.xlsx')}_missing") from error


def _job_object(contents: bytes) -> Mapping[str, object]:
    try:
        value = json.loads(contents)
    except json.JSONDecodeError as error:
        raise EvaluatorError("job_invalid") from error
    if not isinstance(value, dict):
        raise EvaluatorError("job_invalid")
    return cast("dict[str, object]", value)


def _job_string(job: Mapping[str, object], key: str) -> str:
    value = job.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluatorError("job_invalid")
    return value


def _job_cells(job: Mapping[str, object]) -> tuple[str, ...]:
    value = job.get("proxy_cells")
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(cell, str) or not cell for cell in value)
    ):
        raise EvaluatorError("job_invalid")
    return tuple(cast("list[str]", value))


def _job_boolean(job: Mapping[str, object], key: str) -> bool:
    value = job.get(key)
    if not isinstance(value, bool):
        raise EvaluatorError("job_invalid")
    return value


def _sheet_paths(workbook: zipfile.ZipFile) -> dict[str, str]:
    workbook_xml = defusedxml.ElementTree.fromstring(workbook.read("xl/workbook.xml"))
    relationships_xml = defusedxml.ElementTree.fromstring(
        workbook.read("xl/_rels/workbook.xml.rels")
    )
    relationship_targets: dict[str, str] = {}
    for relationship in relationships_xml.findall(f"{{{_PACKAGE_REL_NS}}}Relationship"):
        relationship_id = relationship.get("Id")
        target = relationship.get("Target")
        if relationship_id is None or target is None:
            raise EvaluatorError("workbook_relationship_invalid")
        if target.startswith("/"):
            worksheet_path = posixpath.normpath(target.lstrip("/"))
        else:
            worksheet_path = posixpath.normpath(posixpath.join("xl", target))
        if worksheet_path.startswith("../") or not worksheet_path.startswith("xl/"):
            raise EvaluatorError("workbook_relationship_invalid")
        relationship_targets[relationship_id] = worksheet_path

    sheet_paths: dict[str, str] = {}
    for sheet in workbook_xml.findall(f".//{{{_SPREADSHEET_NS}}}sheet"):
        sheet_name = sheet.get("name")
        relationship_id = sheet.get(f"{{{_DOCUMENT_REL_NS}}}id")
        if sheet_name is None or relationship_id not in relationship_targets:
            raise EvaluatorError("workbook_relationship_invalid")
        sheet_paths[sheet_name] = relationship_targets[relationship_id]
    if not sheet_paths:
        raise EvaluatorError("workbook_has_no_sheets")
    return sheet_paths


def _valid_xlsx(contents: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as workbook:
            names = set(workbook.namelist())
            if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                return False
            sheet_paths = _sheet_paths(workbook)
            if any(sheet_path not in names for sheet_path in sheet_paths.values()):
                return False
            for sheet_path in sheet_paths.values():
                defusedxml.ElementTree.fromstring(workbook.read(sheet_path))
            return True
    except (DefusedXmlException, EvaluatorError, KeyError, ParseError, zipfile.BadZipFile):
        return False


def _workbook_sheet_paths(workbook_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(workbook_path) as workbook:
        return _sheet_paths(workbook)


def _cell_outputs(
    answer_position: str,
    sheet_names: Sequence[str],
    generate_cell_names: Callable[[str], list[str]],
) -> tuple[tuple[str, str], ...]:
    if not sheet_names:
        raise EvaluatorError("reference_has_no_sheets")
    outputs: list[tuple[str, str]] = []
    for raw_position in answer_position.split(","):
        if "!" in raw_position:
            sheet_name, cell_range = raw_position.split("!")
            sheet_name = sheet_name.lstrip("'").rstrip("'")
            cell_range = cell_range.lstrip("'").rstrip("'")
        else:
            sheet_name = sheet_names[0]
            cell_range = raw_position.lstrip("'").rstrip("'")
        outputs.extend((sheet_name, cell_name) for cell_name in generate_cell_names(cell_range))
    return tuple(outputs)


def _proxy_outputs(
    proxy_cells: Sequence[str],
    sheet_names: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    if not sheet_names:
        raise EvaluatorError("reference_has_no_sheets")
    outputs: list[tuple[str, str]] = []
    for position in proxy_cells:
        if "!" in position:
            sheet_name, cell_name = position.split("!")
            outputs.append((sheet_name.lstrip("'").rstrip("'"), cell_name.lstrip("'").rstrip("'")))
        else:
            outputs.append((sheet_names[0], position))
    return tuple(outputs)


def _submitted_output_diagnostics(
    contents: bytes,
    outputs: Sequence[tuple[str, str]],
) -> tuple[dict[str, str | None], ...]:
    diagnostics: list[dict[str, str | None]] = []
    with zipfile.ZipFile(io.BytesIO(contents)) as workbook:
        sheet_paths = _sheet_paths(workbook)
        cells_by_sheet: dict[str, dict[str, Any]] = {}
        for sheet_name, cell_name in dict.fromkeys(outputs):
            sheet_path = sheet_paths.get(sheet_name)
            if sheet_path is None:
                cell = None
            else:
                if sheet_name not in cells_by_sheet:
                    sheet_xml = defusedxml.ElementTree.fromstring(workbook.read(sheet_path))
                    cell_elements: dict[str, Any] = {}
                    for cell_element in sheet_xml.findall(f".//{{{_SPREADSHEET_NS}}}c"):
                        cell_reference = cell_element.get("r")
                        if cell_reference is not None:
                            cell_elements[cell_reference] = cell_element
                    cells_by_sheet[sheet_name] = cell_elements
                cell = cells_by_sheet[sheet_name].get(cell_name)
            formula = None if cell is None else cell.find(f"{{{_SPREADSHEET_NS}}}f")
            diagnostics.append(
                {
                    "sheet_name": sheet_name,
                    "cell_name": cell_name,
                    "formula": None if formula is None else formula.text,
                    "cell_type": None if cell is None else cell.get("t"),
                }
            )
    return tuple(diagnostics)


def _output_formula_state(
    workbook_path: Path,
    outputs: Sequence[tuple[str, str]],
) -> tuple[str | None, frozenset[tuple[str, str]]]:
    sheet_paths = _workbook_sheet_paths(workbook_path)
    requested_by_sheet: dict[str, set[str]] = {}
    for sheet_name, cell_name in outputs:
        requested_by_sheet.setdefault(sheet_name, set()).add(cell_name)

    with zipfile.ZipFile(workbook_path) as workbook:
        error_cells: set[tuple[str, str]] = set()
        for sheet_name, requested_cells in requested_by_sheet.items():
            sheet_path = sheet_paths.get(sheet_name)
            if sheet_path is None:
                continue
            sheet_xml = defusedxml.ElementTree.fromstring(workbook.read(sheet_path))
            for cell in sheet_xml.findall(f".//{{{_SPREADSHEET_NS}}}c"):
                if cell.get("r") not in requested_cells:
                    continue
                formula = cell.find(f"{{{_SPREADSHEET_NS}}}f")
                value = cell.find(f"{{{_SPREADSHEET_NS}}}v")
                if formula is not None and value is None:
                    return "formula_not_recalculated", frozenset(error_cells)
                if cell.get("t") == "e":
                    cell_name = cell.get("r")
                    if cell_name is not None:
                        error_cells.add((sheet_name, cell_name))
    return None, frozenset(error_cells)


def _formula_failure(
    workbook_path: Path,
    outputs: Sequence[tuple[str, str]],
) -> str | None:
    failure, error_cells = _output_formula_state(workbook_path, outputs)
    if failure is not None:
        return failure
    if error_cells:
        return "formula_error"
    return None


def _recalculate(workbook_path: Path, output_directory: Path, profile_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    profile_directory.mkdir(parents=True, exist_ok=True)
    profile_uri = profile_directory.resolve().as_uri()
    try:
        result = subprocess.run(  # noqa: S603
            [
                "/usr/bin/libreoffice",
                f"-env:UserInstallation={profile_uri}",
                "--headless",
                "--calc",
                "--convert-to",
                "xlsx:Calc MS Excel 2007 XML",
                "--outdir",
                str(output_directory),
                str(workbook_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except FileNotFoundError as error:
        raise EvaluatorError("libreoffice_missing") from error
    except subprocess.TimeoutExpired as error:
        raise EvaluatorError("libreoffice_timeout") from error
    if result.returncode != 0:
        raise EvaluatorError("libreoffice_failed")
    recalculated_path = output_directory / workbook_path.name
    if not recalculated_path.is_file():
        raise EvaluatorError("libreoffice_output_missing")
    if not _valid_xlsx(recalculated_path.read_bytes()):
        raise EvaluatorError("libreoffice_output_invalid")
    return recalculated_path


def _load_upstream_evaluator(
    evaluator_path: Path,
) -> tuple[Callable[..., Any], Callable[[str], list[str]]]:
    specification = importlib.util.spec_from_file_location(
        "spreadsheet_upstream_evaluation", evaluator_path
    )
    if specification is None or specification.loader is None:
        raise EvaluatorError("upstream_evaluator_invalid")
    module: ModuleType = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module_namespace = cast("Mapping[str, object]", module.__dict__)
    compare_workbooks = cast("Callable[..., Any]", module_namespace["compare_workbooks"])
    generate_cell_names = cast(
        "Callable[[str], list[str]]", module_namespace["generate_cell_names"]
    )
    return compare_workbooks, generate_cell_names


def _unmeasured(
    kind: str,
    submitted_output_cells: tuple[dict[str, str | None], ...] = (),
) -> dict[str, object]:
    return {
        "status": "unmeasured",
        "proxy": {
            "status": "unmeasured",
            "score": None,
            "success": None,
            "checks_passed": None,
            "checks_total": None,
            "error": {"kind": kind},
        },
        "full": {"status": "unmeasured", "success": None, "error": {"kind": kind}},
        "recalculation": {"status": "failed"},
        "error": {"kind": kind},
        "submitted_output_cells": list(submitted_output_cells),
        "limitations": [_UPSTREAM_LIMITATION],
    }


def _measured_artifact_failure(
    checks_total: int,
    submitted_output_cells: tuple[dict[str, str | None], ...] = (),
) -> dict[str, object]:
    return {
        "status": "measured",
        "proxy": {
            "status": "measured",
            "score": 0.0,
            "success": False,
            "checks_passed": 0,
            "checks_total": checks_total,
            "error": None,
        },
        "full": {"status": "measured", "success": False, "error": None},
        "recalculation": {"status": "not_run_for_invalid_submission"},
        "error": None,
        "submitted_output_cells": list(submitted_output_cells),
        "limitations": [_UPSTREAM_LIMITATION],
    }


def _read_and_check_digest(
    workdir: Path, name: str, job: Mapping[str, object], digest_key: str
) -> bytes:
    contents = _required_bytes(workdir, name)
    expected_digest = _job_string(job, digest_key)
    if _sha256(contents) != expected_digest:
        raise EvaluatorError(f"{digest_key.removesuffix('_sha256')}_digest_mismatch")
    return contents


def _recalculate_assessment_copy(
    *,
    name: str,
    workdir: Path,
    reference_bytes: bytes,
    submission_bytes: bytes,
    cell_outputs: Sequence[tuple[str, str]],
) -> tuple[Path, Path, frozenset[tuple[str, str]]]:
    case_directory = workdir / "assessments" / name
    source_directory = case_directory / "source"
    reference_source = source_directory / "reference.xlsx"
    submission_source = source_directory / "submission.xlsx"
    source_directory.mkdir(parents=True, exist_ok=False)
    reference_source.write_bytes(reference_bytes)
    submission_source.write_bytes(submission_bytes)
    profile_root = case_directory / "profiles"
    reference_path = _recalculate(
        reference_source,
        case_directory / "recalculated-reference",
        profile_root / "reference",
    )
    submission_path = _recalculate(
        submission_source,
        case_directory / "recalculated-submission",
        profile_root / "submission",
    )
    reference_formula_failure = _formula_failure(reference_path, cell_outputs)
    if reference_formula_failure is not None:
        raise EvaluatorError(f"reference_{reference_formula_failure}")
    formula_failure, submission_error_cells = _output_formula_state(submission_path, cell_outputs)
    if formula_failure is not None:
        raise EvaluatorError(formula_failure)
    return reference_path, submission_path, submission_error_cells


def _compare_proxy_cell(
    *,
    reference_path: Path,
    submission_path: Path,
    instruction_type: str,
    cell_position: str,
    compare_workbooks: Callable[..., Any],
) -> bool:
    with contextlib.redirect_stdout(io.StringIO()):
        outcome, _ = compare_workbooks(
            str(reference_path),
            str(submission_path),
            instruction_type,
            cell_position,
        )
    if not isinstance(outcome, bool):
        raise EvaluatorError("upstream_comparator_invalid")
    return outcome


def _compare_full_workbook(
    request: _AssessmentRequest,
    reference_path: Path,
    submission_path: Path,
) -> bool:
    with contextlib.redirect_stdout(io.StringIO()):
        full_success, _ = request.compare_workbooks(
            str(reference_path),
            str(submission_path),
            request.instruction_type,
            request.answer_position,
        )
    if not isinstance(full_success, bool):
        raise EvaluatorError("upstream_comparator_invalid")
    return full_success


def _full_component_result(request: _AssessmentRequest) -> dict[str, object]:
    try:
        reference_path, submission_path, error_cells = _recalculate_assessment_copy(
            name="full",
            workdir=request.evaluation_root,
            reference_bytes=request.reference_bytes,
            submission_bytes=request.submission_bytes,
            cell_outputs=request.full_cells,
        )
        full_success = not error_cells
        if full_success:
            full_success = _compare_full_workbook(request, reference_path, submission_path)
    except EvaluatorError as error:
        return {
            "status": "unmeasured",
            "success": None,
            "error": {"kind": f"full_{error.kind}"},
        }
    return {"status": "measured", "success": full_success, "error": None}


def _proxy_component_result(request: _AssessmentRequest) -> dict[str, object]:
    try:
        reference_path, submission_path, error_cells = _recalculate_assessment_copy(
            name="proxy",
            workdir=request.evaluation_root,
            reference_bytes=request.reference_bytes,
            submission_bytes=request.submission_bytes,
            cell_outputs=request.proxy_outputs,
        )
        proxy_successes = tuple(
            False
            if cell_output in error_cells
            else _compare_proxy_cell(
                reference_path=reference_path,
                submission_path=submission_path,
                instruction_type=request.instruction_type,
                cell_position=cell_position,
                compare_workbooks=request.compare_workbooks,
            )
            for cell_position, cell_output in zip(
                request.proxy_cells, request.proxy_outputs, strict=True
            )
        )
    except EvaluatorError as error:
        return {
            "status": "unmeasured",
            "score": None,
            "success": None,
            "checks_passed": None,
            "checks_total": None,
            "error": {"kind": f"proxy_{error.kind}"},
        }
    proxy_checks_passed = sum(proxy_successes)
    proxy_checks_total = len(proxy_successes)
    return {
        "status": "measured",
        "score": proxy_checks_passed / proxy_checks_total,
        "success": proxy_checks_passed == proxy_checks_total,
        "checks_passed": proxy_checks_passed,
        "checks_total": proxy_checks_total,
        "error": None,
    }


def _assess_workbooks(request: _AssessmentRequest) -> dict[str, object]:
    if not request.operational_only:
        for sheet_name, cell_name in request.proxy_outputs:
            if (sheet_name, cell_name) not in request.full_cells:
                raise EvaluatorError("proxy_position_outside_original_request")

    full_result = (
        {"status": "unmeasured", "success": None, "error": {"kind": "not_requested"}}
        if request.operational_only
        else _full_component_result(request)
    )
    proxy_result = _proxy_component_result(request)
    proxy_status = cast("str", proxy_result["status"])
    return {
        "status": proxy_status,
        "proxy": proxy_result,
        "full": full_result,
        "recalculation": {"status": "succeeded"}
        if proxy_status == "measured" and full_result["status"] == "measured"
        else {"status": "partial"},
        "error": proxy_result.get("error"),
    }


def run_worker(workdir: Path) -> dict[str, object]:
    """Run an evaluation, returning a machine-readable score or explicit error."""
    try:
        return _run_worker(workdir)
    except EvaluatorError as error:
        logger.warning("Spreadsheet evaluation is unmeasured: %s", error.kind)
        return _unmeasured(error.kind)


def _run_worker(workdir: Path) -> dict[str, object]:
    job = _job_object(_required_bytes(workdir, "job.json"))
    input_bytes = _read_and_check_digest(workdir, "input.xlsx", job, "input_sha256")
    reference_bytes = _read_and_check_digest(workdir, "reference.xlsx", job, "reference_sha256")
    submission_bytes = _read_and_check_digest(workdir, "submission.xlsx", job, "submission_sha256")
    evaluator_bytes = _read_and_check_digest(
        workdir,
        "upstream_evaluation.py",
        job,
        "upstream_evaluation_sha256",
    )
    worker_bytes = _read_and_check_digest(
        workdir,
        "spreadsheet_worker.py",
        job,
        "worker_sha256",
    )
    if _sha256(Path(__file__).read_bytes()) != _sha256(worker_bytes):
        raise EvaluatorError("worker_source_mismatch")
    del evaluator_bytes, worker_bytes

    proxy_cells = _job_cells(job)
    instruction_type = _job_string(job, "instruction_type")
    operational_only = _job_boolean(job, "operational_only")
    answer_position = "" if operational_only else _job_string(job, "answer_position")
    if not _valid_xlsx(input_bytes) or not _valid_xlsx(reference_bytes):
        raise EvaluatorError("source_workbook_invalid")
    if not _valid_xlsx(submission_bytes):
        if operational_only:
            result = _measured_artifact_failure(len(proxy_cells))
            result["full"] = {
                "status": "unmeasured",
                "success": None,
                "error": {"kind": "not_requested"},
            }
            return result
        return _measured_artifact_failure(len(proxy_cells))

    evaluator_path = workdir / "upstream_evaluation.py"
    compare_workbooks, generate_cell_names = _load_upstream_evaluator(evaluator_path)
    with tempfile.TemporaryDirectory(prefix="spreadsheet-worker-", dir=workdir) as temp_directory:
        evaluation_root = Path(temp_directory)
        original_reference = evaluation_root / "original-reference.xlsx"
        original_reference.write_bytes(reference_bytes)
        reference_sheet_names = tuple(_workbook_sheet_paths(original_reference))
        proxy_outputs = _proxy_outputs(proxy_cells, reference_sheet_names)
        full_cells = (
            ()
            if operational_only
            else _cell_outputs(answer_position, reference_sheet_names, generate_cell_names)
        )
        diagnostic_outputs = proxy_outputs if operational_only else full_cells
        submitted_output_cells = _submitted_output_diagnostics(submission_bytes, diagnostic_outputs)
        request = _AssessmentRequest(
            evaluation_root=evaluation_root,
            reference_bytes=reference_bytes,
            submission_bytes=submission_bytes,
            full_cells=full_cells,
            proxy_outputs=proxy_outputs,
            proxy_cells=proxy_cells,
            instruction_type=instruction_type,
            answer_position=answer_position,
            compare_workbooks=compare_workbooks,
            operational_only=operational_only,
        )
        try:
            result = _assess_workbooks(request)
        except EvaluatorError as error:
            return _unmeasured(error.kind, submitted_output_cells)
        result["submitted_output_cells"] = list(submitted_output_cells)
        result["limitations"] = [_UPSTREAM_LIMITATION]
        return result


def main() -> int:
    """Run from the trusted container work directory and print machine output."""
    logging.basicConfig(level=logging.INFO)
    result = run_worker(Path.cwd())
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
