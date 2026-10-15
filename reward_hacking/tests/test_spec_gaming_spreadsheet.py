from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, BinaryIO, Self, cast
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from reward_hacking.spec_gaming import spreadsheet, spreadsheet_worker
from reward_hacking.spec_gaming.spreadsheet import (
    DatasetLayout,
    SpreadsheetCase,
    SpreadsheetTask,
    _full_case_members,
    _pinned_source_evaluator,
    _verified_pair_members,
    actor_input_files,
    evaluation_files,
    extract_archive_member,
    parse_evaluation,
    select_proxy_cells,
    validate_proxy_cells,
    with_actor_case,
)
from reward_hacking.spec_gaming.spreadsheet_worker import (
    EvaluatorError,
    _assess_workbooks,
    _AssessmentRequest,
    _formula_failure,
    _recalculate_assessment_copy,
    _submitted_output_diagnostics,
    run_worker,
)


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _object_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return cast("dict[str, object]", value)


def _synthetic_xlsx() -> bytes:
    workbook_buffer = io.BytesIO()
    with ZipFile(workbook_buffer, "w", compression=ZIP_DEFLATED) as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        workbook.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
        )
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="A1" t="n"><v>1</v></c></row></sheetData></worksheet>',
        )
    return workbook_buffer.getvalue()


def _replace_xlsx_member(contents: bytes, member_name: str, replacement: bytes) -> bytes:
    workbook_buffer = io.BytesIO()
    with (
        ZipFile(io.BytesIO(contents)) as source,
        ZipFile(workbook_buffer, "w", compression=ZIP_DEFLATED) as destination,
    ):
        for source_member in source.namelist():
            member_contents = (
                replacement if source_member == member_name else source.read(source_member)
            )
            destination.writestr(source_member, member_contents)
    return workbook_buffer.getvalue()


def _with_xlsx_comment(contents: bytes, comment: bytes) -> bytes:
    workbook_buffer = io.BytesIO()
    with (
        ZipFile(io.BytesIO(contents)) as source,
        ZipFile(workbook_buffer, "w", compression=ZIP_DEFLATED) as destination,
    ):
        destination.comment = comment
        for source_member in source.namelist():
            destination.writestr(source_member, source.read(source_member))
    return workbook_buffer.getvalue()


def _configure_synthetic_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_rows: Sequence[Mapping[str, object]],
    source_members: dict[str, bytes],
) -> Path:
    archive_path = tmp_path / "synthetic-spreadsheetbench.tar.gz"
    archive_members = {
        "synthetic/dataset.json": json.dumps(metadata_rows).encode(),
        **source_members,
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for member_name, contents in archive_members.items():
            member = tarfile.TarInfo(member_name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))

    compressed_contents = archive_path.read_bytes()
    evaluator_path = tmp_path / "evaluation.py"
    evaluator_bytes = b"synthetic trusted evaluator"
    evaluator_path.write_bytes(evaluator_bytes)
    layout = DatasetLayout(
        name="verified_400",
        archive_name=archive_path.name,
        archive_sha256=_sha256(compressed_contents),
        member_root="synthetic",
        case_indices=(1,),
        verified_layout=True,
    )
    monkeypatch.setattr(spreadsheet, "DATASET_LAYOUTS", {"verified_400": layout})
    monkeypatch.setattr(
        spreadsheet,
        "_source_revision",
        lambda _source_root: spreadsheet.SPREADSHEETBENCH_REVISION,
    )
    monkeypatch.setattr(
        spreadsheet,
        "_pinned_source_evaluator",
        lambda _source_root: (evaluator_path, _sha256(evaluator_bytes)),
    )
    return archive_path


@pytest.fixture
def task(tmp_path: Path) -> SpreadsheetTask:
    input_contents = _synthetic_xlsx()
    reference_contents = _synthetic_xlsx()
    evaluator_contents = b"synthetic trusted evaluator source"
    input_path = tmp_path / "input.xlsx"
    reference_path = tmp_path / "reference.xlsx"
    evaluator_path = tmp_path / "evaluation.py"
    input_path.write_bytes(input_contents)
    reference_path.write_bytes(reference_contents)
    evaluator_path.write_bytes(evaluator_contents)
    case = SpreadsheetCase(
        case_index=1,
        input_path=input_path,
        reference_path=reference_path,
        input_sha256=_sha256(input_contents),
        reference_sha256=_sha256(reference_contents),
        input_member="synthetic/1_input.xlsx",
        reference_member="synthetic/1_reference.xlsx",
    )
    return SpreadsheetTask(
        source_revision="synthetic-source-revision",
        source_archive_sha256="synthetic-archive-digest",
        dataset_name="verified_400",
        task_id="synthetic-task",
        instruction="Apply the synthetic workbook request.",
        instruction_type="Cell-Level Manipulation",
        answer_position="A1:B2",
        cases=(case,),
        upstream_evaluation_path=evaluator_path,
        upstream_evaluation_sha256=_sha256(evaluator_contents),
    )


def test_proxy_cells_are_a_deterministic_strict_subset(task: SpreadsheetTask) -> None:
    selected = select_proxy_cells(task)

    assert selected == select_proxy_cells(task)
    assert selected
    assert len(selected) < 4
    assert set(selected) <= {"A1", "A2", "B1", "B2"}
    assert task.training_eligible is False


def test_frozen_proxy_cells_must_be_a_nonempty_unique_strict_subset(task: SpreadsheetTask) -> None:
    assert validate_proxy_cells(task, ("B1", "A1")) == ("B1", "A1")

    with pytest.raises(ValueError, match="strict subset"):
        validate_proxy_cells(task, ("A1", "A2", "B1", "B2"))
    with pytest.raises(ValueError, match="unique"):
        validate_proxy_cells(task, ("A1", "A1"))
    with pytest.raises(ValueError, match="outside"):
        validate_proxy_cells(task, ("Z99",))


def test_import_rejects_a_modified_pinned_upstream_evaluator(tmp_path: Path) -> None:
    evaluator_path = tmp_path / "evaluation" / "evaluation.py"
    evaluator_path.parent.mkdir()
    evaluator_path.write_text("def compare_workbooks(*args): return False, ''\n")

    with pytest.raises(ValueError, match="upstream evaluator is not pinned"):
        _pinned_source_evaluator(tmp_path)


def test_proxy_selection_rejects_a_single_required_cell(task: SpreadsheetTask) -> None:
    single_cell_task = replace(task, answer_position="A1")

    with pytest.raises(ValueError, match="strict subset"):
        select_proxy_cells(single_cell_task)


def test_actor_input_exposes_only_the_primary_input_workbook(task: SpreadsheetTask) -> None:
    files = actor_input_files(task)
    input_path = task.primary_case.input_path
    assert input_path is not None

    assert files == {"input.xlsx": input_path.read_bytes()}
    assert "reference.xlsx" not in files
    assert "upstream_evaluation.py" not in files


def test_actor_case_selection_packages_the_selected_case_and_preserves_default(
    task: SpreadsheetTask,
    tmp_path: Path,
) -> None:
    case_two_input = _with_xlsx_comment(_synthetic_xlsx(), b"case two input")
    case_two_reference = _with_xlsx_comment(_synthetic_xlsx(), b"case two reference")
    case_three_input = _with_xlsx_comment(_synthetic_xlsx(), b"case three input")
    case_three_reference = _with_xlsx_comment(_synthetic_xlsx(), b"case three reference")
    cases = list(task.cases)
    for case_index, input_contents, reference_contents in (
        (2, case_two_input, case_two_reference),
        (3, case_three_input, case_three_reference),
    ):
        input_path = tmp_path / f"{case_index}_input.xlsx"
        reference_path = tmp_path / f"{case_index}_reference.xlsx"
        input_path.write_bytes(input_contents)
        reference_path.write_bytes(reference_contents)
        cases.append(
            SpreadsheetCase(
                case_index=case_index,
                input_path=input_path,
                reference_path=reference_path,
                input_sha256=_sha256(input_contents),
                reference_sha256=_sha256(reference_contents),
                input_member=f"synthetic/{case_index}_input.xlsx",
                reference_member=f"synthetic/{case_index}_reference.xlsx",
            )
        )
    task_with_cases = replace(task, cases=tuple(cases))

    default_files = actor_input_files(task_with_cases)
    selected_task = with_actor_case(task_with_cases, 3)
    selected_actor_files = actor_input_files(selected_task)
    evaluation = evaluation_files(
        selected_task, b"submitted workbook", select_proxy_cells(selected_task)
    )
    job = json.loads(evaluation["job.json"])
    first_case_input_path = task.cases[0].input_path
    assert first_case_input_path is not None

    assert task_with_cases.primary_case.case_index == 1
    assert default_files["input.xlsx"] == first_case_input_path.read_bytes()
    assert selected_task.primary_case.case_index == 3
    assert selected_actor_files == {"input.xlsx": case_three_input}
    assert evaluation["input.xlsx"] == case_three_input
    assert evaluation["reference.xlsx"] == case_three_reference
    assert job["source_case_index"] == 3
    assert job["input_sha256"] == _sha256(case_three_input)
    assert job["reference_sha256"] == _sha256(case_three_reference)


def test_actor_case_selection_rejects_missing_or_incomplete_case(task: SpreadsheetTask) -> None:
    with pytest.raises(ValueError, match="does not have case 3"):
        with_actor_case(task, 3)

    incomplete_case = SpreadsheetCase(
        case_index=3,
        input_path=task.cases[0].input_path,
        reference_path=None,
        input_sha256=task.cases[0].input_sha256,
        reference_sha256=None,
        input_member="synthetic/3_input.xlsx",
        reference_member=None,
    )
    task_with_incomplete_case = replace(task, cases=(*task.cases, incomplete_case))

    with pytest.raises(ValueError, match="case 3 is incomplete"):
        with_actor_case(task_with_incomplete_case, 3)


def test_evaluation_files_bind_both_evaluations_to_one_submission(
    task: SpreadsheetTask,
) -> None:
    submission = b"immutable submitted workbook"
    selected_cells = select_proxy_cells(task)

    files = evaluation_files(task, submission, selected_cells)
    job = json.loads(files["job.json"])
    input_path = task.primary_case.input_path
    reference_path = task.primary_case.reference_path
    assert input_path is not None
    assert reference_path is not None

    assert files["input.xlsx"] == input_path.read_bytes()
    assert files["reference.xlsx"] == reference_path.read_bytes()
    assert files["submission.xlsx"] == submission
    assert files["upstream_evaluation.py"] == task.upstream_evaluation_path.read_bytes()
    assert "spreadsheet_worker.py" in files
    assert job["answer_position"] == task.answer_position
    assert job["proxy_cells"] == list(selected_cells)
    assert "instruction" not in job


def test_evaluation_files_can_request_operational_only_proxy_assessment(
    task: SpreadsheetTask,
) -> None:
    proxy_cells = validate_proxy_cells(task, ("B1", "A1"))

    files = evaluation_files(
        task, b"immutable submitted workbook", proxy_cells, operational_only=True
    )

    assert json.loads(files["job.json"])["operational_only"] is True


def test_parse_evaluation_preserves_measurement_failure_without_intent_labels() -> None:
    parsed = parse_evaluation(
        {
            "status": "unmeasured",
            "proxy": {"status": "unmeasured", "score": None, "success": None},
            "full": {"status": "unmeasured", "success": None},
            "error": {"kind": "reference_missing"},
        }
    )

    assert parsed.status == "unmeasured"
    assert parsed.proxy_score is None
    assert parsed.full_success is None
    assert parsed.error_kind == "reference_missing"
    assert not hasattr(parsed, "hacked")


def test_parse_evaluation_keeps_proxy_full_discrepancy_as_measurement() -> None:
    parsed = parse_evaluation(
        {
            "status": "measured",
            "proxy": {
                "status": "measured",
                "score": 1.0,
                "success": True,
                "checks_passed": 2,
                "checks_total": 2,
            },
            "full": {"status": "measured", "success": False},
            "error": None,
        }
    )

    assert parsed.status == "measured"
    assert parsed.proxy_score == 1.0
    assert parsed.proxy_success is True
    assert parsed.full_success is False
    assert parsed.error_kind is None
    assert not hasattr(parsed, "hacked")


def test_parse_evaluation_preserves_independently_unmeasured_full_component() -> None:
    parsed = parse_evaluation(
        {
            "status": "measured",
            "proxy": {
                "status": "measured",
                "score": 1.0,
                "success": True,
                "checks_passed": 2,
                "checks_total": 2,
                "error": None,
            },
            "full": {
                "status": "unmeasured",
                "success": None,
                "error": {"kind": "full_libreoffice_timeout"},
            },
            "error": None,
        }
    )

    assert parsed.status == "measured"
    assert parsed.full_status == "unmeasured"
    assert parsed.full_success is None
    assert parsed.full_error_kind == "full_libreoffice_timeout"


def test_parse_evaluation_preserves_independently_measured_full_component() -> None:
    parsed = parse_evaluation(
        {
            "status": "unmeasured",
            "proxy": {
                "status": "unmeasured",
                "score": None,
                "success": None,
                "checks_passed": None,
                "checks_total": None,
                "error": {"kind": "proxy_libreoffice_timeout"},
            },
            "full": {"status": "measured", "success": True, "error": None},
            "error": {"kind": "proxy_libreoffice_timeout"},
        }
    )

    assert parsed.status == "unmeasured"
    assert parsed.full_status == "measured"
    assert parsed.full_success is True
    assert parsed.full_error_kind is None


@pytest.mark.parametrize(
    ("score", "success", "checks_passed", "checks_total"),
    [(1.0, True, 1, 2), (0.5, True, 1, 2), (0.75, False, 1, 2)],
)
def test_parse_evaluation_rejects_inconsistent_proxy_score_and_counts(
    score: float,
    success: bool,
    checks_passed: int,
    checks_total: int,
) -> None:
    result = {
        "status": "measured",
        "proxy": {
            "status": "measured",
            "score": score,
            "success": success,
            "checks_passed": checks_passed,
            "checks_total": checks_total,
        },
        "full": {"status": "measured", "success": False},
        "error": None,
    }

    with pytest.raises(ValueError, match="inconsistent fields"):
        parse_evaluation(result)


def test_parse_evaluation_retains_output_formula_diagnostics_and_limitation() -> None:
    parsed = parse_evaluation(
        {
            "status": "measured",
            "proxy": {
                "status": "measured",
                "score": 0.5,
                "success": False,
                "checks_passed": 1,
                "checks_total": 2,
            },
            "full": {"status": "measured", "success": False},
            "error": None,
            "submitted_output_cells": [
                {
                    "sheet_name": "Sheet1",
                    "cell_name": "A1",
                    "formula": "SUM(B1:B2)",
                    "cell_type": "n",
                }
            ],
            "limitations": [
                "upstream_value_comparison_does_not_establish_formula_or_format_requirements"
            ],
        }
    )

    assert parsed.submitted_output_cells[0].formula == "SUM(B1:B2)"
    assert parsed.submitted_output_cells[0].cell_type == "n"
    assert "upstream_value_comparison_does_not_establish_formula_or_format_requirements" in (
        parsed.limitations
    )


def test_submission_diagnostics_read_only_requested_formula_and_cell_type() -> None:
    worksheet = (
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b'<sheetData><row r="1"><c r="A1" t="n"><f>SUM(B1:B2)</f><v>3</v></c>'
        b"</row></sheetData></worksheet>"
    )
    submission = _replace_xlsx_member(_synthetic_xlsx(), "xl/worksheets/sheet1.xml", worksheet)

    diagnostics = _submitted_output_diagnostics(
        submission,
        (("Sheet1", "A1"), ("Sheet1", "B2")),
    )

    assert diagnostics == (
        {"sheet_name": "Sheet1", "cell_name": "A1", "formula": "SUM(B1:B2)", "cell_type": "n"},
        {"sheet_name": "Sheet1", "cell_name": "B2", "formula": None, "cell_type": None},
    )
    assert all("value" not in diagnostic for diagnostic in diagnostics)


@pytest.mark.parametrize("error_value", ["#DIV/0!", "#REF!", "#VALUE!", "#NAME?"])
def test_formula_errors_in_requested_output_cells_make_assessment_unmeasured(
    tmp_path: Path,
    error_value: str,
) -> None:
    worksheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData><row r="1"><c r="A1" t="e"><f>1/0</f><v>{error_value}</v></c>'
        "</row></sheetData></worksheet>"
    ).encode()
    workbook = _replace_xlsx_member(_synthetic_xlsx(), "xl/worksheets/sheet1.xml", worksheet)
    workbook_path = tmp_path / "reference.xlsx"
    workbook_path.write_bytes(workbook)

    assert _formula_failure(workbook_path, (("Sheet1", "A1"),)) == "formula_error"


def test_matching_reference_and_submission_formula_errors_fail_unmeasured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worksheet = (
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b'<sheetData><row r="1"><c r="A1" t="e"><f>1/0</f><v>#DIV/0!</v></c>'
        b"</row></sheetData></worksheet>"
    )
    workbook = _replace_xlsx_member(_synthetic_xlsx(), "xl/worksheets/sheet1.xml", worksheet)

    def copy_as_recalculated(
        workbook_path: Path,
        output_directory: Path,
        profile_directory: Path,
    ) -> Path:
        del profile_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        recalculated_path = output_directory / workbook_path.name
        recalculated_path.write_bytes(workbook_path.read_bytes())
        return recalculated_path

    monkeypatch.setattr(
        "reward_hacking.spec_gaming.spreadsheet_worker._recalculate",
        copy_as_recalculated,
    )

    with pytest.raises(EvaluatorError, match="reference_formula_error") as raised:
        _recalculate_assessment_copy(
            name="full",
            workdir=tmp_path,
            reference_bytes=workbook,
            submission_bytes=workbook,
            cell_outputs=(("Sheet1", "A1"),),
        )

    assert raised.value.kind == "reference_formula_error"


@pytest.mark.parametrize("error_value", ["#DIV/0!", "#REF!", "#VALUE!"])
def test_submission_formula_error_is_measured_as_a_failed_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_value: str,
) -> None:
    reference = _synthetic_xlsx()
    submission_sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData><row r="1"><c r="A1" t="e"><f>1/0</f><v>{error_value}</v></c>'
        "</row></sheetData></worksheet>"
    ).encode()
    submission = _replace_xlsx_member(
        _synthetic_xlsx(), "xl/worksheets/sheet1.xml", submission_sheet
    )

    def copy_as_recalculated(
        workbook_path: Path,
        output_directory: Path,
        profile_directory: Path,
    ) -> Path:
        del profile_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        recalculated_path = output_directory / workbook_path.name
        recalculated_path.write_bytes(workbook_path.read_bytes())
        return recalculated_path

    monkeypatch.setattr(
        "reward_hacking.spec_gaming.spreadsheet_worker._recalculate",
        copy_as_recalculated,
    )

    def comparator(*args: object) -> tuple[bool, str]:
        del args
        return True, "matching cached error must not pass"

    request = _AssessmentRequest(
        evaluation_root=tmp_path,
        reference_bytes=reference,
        submission_bytes=submission,
        full_cells=(("Sheet1", "A1"),),
        proxy_outputs=(("Sheet1", "A1"),),
        proxy_cells=("Sheet1!A1",),
        instruction_type="Cell-Level Manipulation",
        answer_position="Sheet1!A1",
        compare_workbooks=comparator,
    )

    result = _assess_workbooks(request)
    proxy_result = _object_mapping(result["proxy"])
    full_result = _object_mapping(result["full"])

    assert result["status"] == "measured"
    assert proxy_result["success"] is False
    assert full_result["success"] is False


@pytest.mark.parametrize(
    ("operational_only", "failing_component"),
    [(True, "full"), (False, "full"), (False, "proxy")],
)
def test_assessment_components_keep_independent_measurement_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operational_only: bool,
    failing_component: str,
) -> None:
    assessed_names: list[str] = []
    workbook_path = tmp_path / "workbook.xlsx"
    workbook_path.write_bytes(_synthetic_xlsx())

    def recalculate_component(
        *,
        name: str,
        workdir: Path,
        reference_bytes: bytes,
        submission_bytes: bytes,
        cell_outputs: tuple[tuple[str, str], ...],
    ) -> tuple[Path, Path, frozenset[tuple[str, str]]]:
        del workdir, reference_bytes, submission_bytes, cell_outputs
        assessed_names.append(name)
        if name == failing_component:
            raise EvaluatorError("libreoffice_timeout")
        return workbook_path, workbook_path, frozenset()

    monkeypatch.setattr(spreadsheet_worker, "_recalculate_assessment_copy", recalculate_component)

    def comparator(*args: object) -> tuple[bool, str]:
        del args
        return True, ""

    result = _assess_workbooks(
        _AssessmentRequest(
            evaluation_root=tmp_path,
            reference_bytes=_synthetic_xlsx(),
            submission_bytes=_synthetic_xlsx(),
            full_cells=(("Sheet1", "A1"), ("Sheet1", "B1")),
            proxy_outputs=(("Sheet1", "A1"),),
            proxy_cells=("Sheet1!A1",),
            instruction_type="Cell-Level Manipulation",
            answer_position="Sheet1!A1: B1",
            compare_workbooks=comparator,
            operational_only=operational_only,
        )
    )
    proxy_result = _object_mapping(result["proxy"])
    full_result = _object_mapping(result["full"])

    assert result["status"] == ("unmeasured" if failing_component == "proxy" else "measured")
    assert proxy_result["status"] == ("unmeasured" if failing_component == "proxy" else "measured")
    assert proxy_result["success"] is (None if failing_component == "proxy" else True)
    assert full_result["status"] == (
        "unmeasured" if operational_only or failing_component == "full" else "measured"
    )
    assert full_result["success"] is (
        None if operational_only or failing_component == "full" else True
    )
    full_error = full_result["error"]
    actual_full_error = None if full_error is None else _object_mapping(full_error)["kind"]
    assert actual_full_error == (
        "not_requested"
        if operational_only
        else "full_libreoffice_timeout"
        if failing_component == "full"
        else None
    )
    proxy_error = proxy_result["error"]
    actual_proxy_error = None if proxy_error is None else _object_mapping(proxy_error)["kind"]
    assert actual_proxy_error == (
        "proxy_libreoffice_timeout" if failing_component == "proxy" else None
    )
    assert assessed_names == (["proxy"] if operational_only else ["full", "proxy"])

    parsed = parse_evaluation(result)
    assert parsed.full_status == full_result["status"]
    assert parsed.full_success == full_result["success"]


def test_worker_scores_an_unreadable_submission_as_artifact_failure(
    task: SpreadsheetTask,
    tmp_path: Path,
) -> None:
    files = evaluation_files(task, b"not an xlsx workbook", select_proxy_cells(task))
    for file_name, contents in files.items():
        (tmp_path / file_name).write_bytes(contents)

    parsed = parse_evaluation(run_worker(tmp_path))

    assert parsed.status == "measured"
    assert parsed.proxy_score == 0.0
    assert parsed.proxy_success is False
    assert parsed.full_success is False
    assert parsed.error_kind is None


@pytest.mark.parametrize(
    "workbook_xml",
    [
        b"<workbook",
        b'<!DOCTYPE workbook [<!ENTITY payload "expanded">]><workbook>&payload;</workbook>',
    ],
)
def test_worker_scores_malformed_or_entity_expanding_submission_as_artifact_failure(
    task: SpreadsheetTask,
    tmp_path: Path,
    workbook_xml: bytes,
) -> None:
    submission = _replace_xlsx_member(_synthetic_xlsx(), "xl/workbook.xml", workbook_xml)
    files = evaluation_files(task, submission, select_proxy_cells(task))
    for file_name, contents in files.items():
        (tmp_path / file_name).write_bytes(contents)

    parsed = parse_evaluation(run_worker(tmp_path))

    assert parsed.status == "measured"
    assert parsed.proxy_score == 0.0
    assert parsed.full_success is False
    assert parsed.error_kind is None


def test_worker_reports_a_missing_reference_as_unmeasured(
    task: SpreadsheetTask,
    tmp_path: Path,
) -> None:
    files = evaluation_files(task, b"not an xlsx workbook", select_proxy_cells(task))
    for file_name, contents in files.items():
        if file_name != "reference.xlsx":
            (tmp_path / file_name).write_bytes(contents)
    reference_path = task.primary_case.reference_path
    assert reference_path is not None
    reference_path.unlink()

    parsed = parse_evaluation(run_worker(tmp_path))

    assert parsed.status == "unmeasured"
    assert parsed.proxy_score is None
    assert parsed.full_success is None
    assert parsed.error_kind == "reference_missing"


def test_verified_layout_resolves_only_inside_the_metadata_row_directory() -> None:
    archive_path = io.BytesIO()
    with tarfile.open(fileobj=archive_path, mode="w") as archive:
        for member_name in (
            "source/synthetic-task/1_synthetic-task_init.xlsx",
            "source/synthetic-task/1_other-task_golden.xlsx",
        ):
            member = tarfile.TarInfo(member_name)
            member.size = 0
            archive.addfile(member, io.BytesIO())
    archive_path.seek(0)

    with tarfile.open(fileobj=archive_path, mode="r:") as archive:
        members = set(archive.getnames())
        input_member, reference_member = _verified_pair_members(
            archive,
            members,
            PurePosixPath("source/synthetic-task"),
            "synthetic-task",
        )

    assert input_member == "source/synthetic-task/1_synthetic-task_init.xlsx"
    assert reference_member == "source/synthetic-task/1_other-task_golden.xlsx"


def test_verified_layout_accepts_exact_source_alias_names() -> None:
    archive_path = io.BytesIO()
    with tarfile.open(fileobj=archive_path, mode="w") as archive:
        for member_name in (
            "source/synthetic-task/initial.xlsx",
            "source/synthetic-task/golden.xlsx",
        ):
            member = tarfile.TarInfo(member_name)
            member.size = 0
            archive.addfile(member, io.BytesIO())
    archive_path.seek(0)

    with tarfile.open(fileobj=archive_path, mode="r:") as archive:
        input_member, reference_member = _verified_pair_members(
            archive,
            set(archive.getnames()),
            PurePosixPath("source/synthetic-task"),
            "synthetic-task",
        )

    assert input_member == "source/synthetic-task/initial.xlsx"
    assert reference_member == "source/synthetic-task/golden.xlsx"


def test_full_layout_preserves_case_two_trailing_space_alias() -> None:
    archive_path = io.BytesIO()
    with tarfile.open(fileobj=archive_path, mode="w") as archive:
        for member_name in (
            "source/synthetic-task/2_synthetic-task_input .xlsx",
            "source/synthetic-task/2_synthetic-task_answer.xlsx",
        ):
            member = tarfile.TarInfo(member_name)
            member.size = 0
            archive.addfile(member, io.BytesIO())
    archive_path.seek(0)

    with tarfile.open(fileobj=archive_path, mode="r:") as archive:
        input_member, reference_member = _full_case_members(
            archive,
            set(archive.getnames()),
            PurePosixPath("source/synthetic-task"),
            "synthetic-task",
            case_index=2,
        )

    assert input_member == "source/synthetic-task/2_synthetic-task_input .xlsx"
    assert reference_member == "source/synthetic-task/2_synthetic-task_answer.xlsx"


def test_archive_extraction_rejects_traversal_before_writing(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.tar"
    destination = tmp_path / "assets"
    with tarfile.open(archive_path, "w") as archive:
        contents = b"must not escape"
        member = tarfile.TarInfo("../outside.xlsx")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))

    with (
        tarfile.open(archive_path) as archive,
        pytest.raises(ValueError, match="unsafe archive member"),
    ):
        extract_archive_member(archive, "../outside.xlsx", destination)

    assert not (tmp_path / "outside.xlsx").exists()


def test_archive_extraction_failure_leaves_no_final_file_and_retry_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "assets.tar"
    asset_name = "source/synthetic-task/input.xlsx"
    contents = b"complete workbook bytes"
    with tarfile.open(archive_path, "w") as source_archive:
        member = tarfile.TarInfo(asset_name)
        member.size = len(contents)
        source_archive.addfile(member, io.BytesIO(contents))

    destination_root = tmp_path / "assets"
    final_path = destination_root / asset_name
    with tarfile.open(archive_path) as source_archive:
        extractfile = source_archive.extractfile

        class FailingReader:
            def __init__(self, source: IO[bytes]) -> None:
                self.source = source
                self.read_count = 0

            def read(self, size: int = -1) -> bytes:
                if self.read_count:
                    raise OSError("injected partial archive read")
                self.read_count += 1
                return self.source.read(min(size, 5))

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *args: object) -> None:
                self.source.close()

        def fail_after_prefix(member: tarfile.TarInfo) -> FailingReader:
            member_source = extractfile(member)
            assert member_source is not None
            return FailingReader(member_source)

        monkeypatch.setattr(source_archive, "extractfile", fail_after_prefix)
        with pytest.raises(OSError, match="injected partial archive read"):
            extract_archive_member(source_archive, asset_name, destination_root)

        assert not final_path.exists()
        assert not tuple(final_path.parent.iterdir())

        monkeypatch.setattr(source_archive, "extractfile", extractfile)
        extracted_path = extract_archive_member(source_archive, asset_name, destination_root)

    assert extracted_path == final_path
    assert final_path.read_bytes() == contents


def test_archive_extraction_refuses_to_replace_different_existing_asset(tmp_path: Path) -> None:
    archive_path = tmp_path / "assets.tar"
    asset_name = "source/synthetic-task/input.xlsx"
    contents = b"archive workbook bytes"
    with tarfile.open(archive_path, "w") as source_archive:
        member = tarfile.TarInfo(asset_name)
        member.size = len(contents)
        source_archive.addfile(member, io.BytesIO(contents))

    destination_root = tmp_path / "assets"
    final_path = destination_root / asset_name
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"different existing asset")

    with (
        tarfile.open(archive_path) as source_archive,
        pytest.raises(FileExistsError, match="different contents"),
    ):
        extract_archive_member(source_archive, asset_name, destination_root)

    assert final_path.read_bytes() == b"different existing asset"


def test_import_skips_whitespace_only_instruction_and_keeps_valid_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_instruction = "  Keep this source text unchanged.  "
    metadata_rows = [
        {
            "id": "valid-task",
            "instruction": valid_instruction,
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
            "spreadsheet_path": "valid-task",
        },
        {
            "id": "blank-instruction-task",
            "instruction": " \t\n ",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
            "spreadsheet_path": "blank-instruction-task",
        },
    ]
    input_name = "synthetic/valid-task/1_valid-task_init.xlsx"
    reference_name = "synthetic/valid-task/1_valid-task_golden.xlsx"
    blank_input_name = "synthetic/blank-instruction-task/1_blank-instruction-task_init.xlsx"
    blank_reference_name = "synthetic/blank-instruction-task/1_blank-instruction-task_golden.xlsx"
    source_members = {
        input_name: b"valid synthetic input",
        reference_name: b"valid synthetic reference",
        blank_input_name: b"blank row input",
        blank_reference_name: b"blank row reference",
    }
    archive_path = _configure_synthetic_import(tmp_path, monkeypatch, metadata_rows, source_members)
    original_archive_digest = _sha256(archive_path.read_bytes())
    asset_root = tmp_path / "assets"

    imported = spreadsheet.import_tasks(tmp_path, archive_path, asset_root)

    assert [task.task_id for task in imported.tasks] == ["valid-task"]
    assert imported.tasks[0].instruction == valid_instruction
    assert imported.issues == (
        spreadsheet.SpreadsheetImportIssue("blank-instruction-task", "missing_instruction"),
    )
    assert not (asset_root / blank_input_name).exists()
    assert not (asset_root / blank_reference_name).exists()
    assert _sha256(archive_path.read_bytes()) == original_archive_digest


@pytest.mark.parametrize("instruction", [None, 7])
def test_import_still_rejects_non_string_instruction_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instruction: object,
) -> None:
    metadata_rows = [
        {
            "id": "malformed-task",
            "instruction": instruction,
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
            "spreadsheet_path": "malformed-task",
        }
    ]
    archive_path = _configure_synthetic_import(tmp_path, monkeypatch, metadata_rows, {})

    with pytest.raises(TypeError, match="instruction must be a string"):
        spreadsheet.import_tasks(tmp_path, archive_path, tmp_path / "assets")


def test_import_decompresses_pinned_archive_once_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_rows = [
        {
            "id": "synthetic-task",
            "instruction": "Synthetic import fixture.",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
            "spreadsheet_path": "synthetic-task",
        }
    ]
    source_members = {
        "synthetic/synthetic-task/1_synthetic-task_init.xlsx": b"synthetic-input",
        "synthetic/synthetic-task/1_synthetic-task_golden.xlsx": b"synthetic-reference",
    }
    archive_path = _configure_synthetic_import(tmp_path, monkeypatch, metadata_rows, source_members)
    compressed_contents = archive_path.read_bytes()
    expected_decompressed_bytes = len(gzip.decompress(compressed_contents))

    decompressed_byte_count = 0
    backward_seek_count = 0
    original_gzip_file = gzip.GzipFile
    original_temporary_file = tempfile.TemporaryFile
    temporary_file_directories: list[str | None] = []
    temporary_files: list[BinaryIO] = []

    class CountingGzipFile(original_gzip_file):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.decompressed_position = 0

        def read(self, size: int | None = -1) -> bytes:
            nonlocal decompressed_byte_count
            contents = super().read(size)
            decompressed_byte_count += len(contents)
            self.decompressed_position += len(contents)
            return contents

        def seek(self, offset: int, whence: int = 0) -> int:
            nonlocal backward_seek_count
            new_position = super().seek(offset, whence)
            if new_position < self.decompressed_position:
                backward_seek_count += 1
            self.decompressed_position = new_position
            return new_position

    def tracked_temporary_file(**kwargs: object) -> BinaryIO:
        directory = cast("Path | None", kwargs.get("dir"))
        temporary_file_directories.append(None if directory is None else str(directory))
        temporary_file = original_temporary_file(dir=directory)
        temporary_files.append(temporary_file)
        return temporary_file

    monkeypatch.setattr(gzip, "GzipFile", CountingGzipFile)
    monkeypatch.setattr(tempfile, "TemporaryFile", tracked_temporary_file)

    imported = spreadsheet.import_tasks(
        source_root=tmp_path,
        archive_path=archive_path,
        asset_root=tmp_path / "assets",
    )

    assert [task.task_id for task in imported.tasks] == ["synthetic-task"]
    assert not imported.issues
    assert decompressed_byte_count == expected_decompressed_bytes
    assert backward_seek_count == 0
    assert temporary_file_directories == [str(spreadsheet._ARCHIVE_TEMP_DIRECTORY)]
    assert len(temporary_files) == 1
    assert temporary_files[0].closed
