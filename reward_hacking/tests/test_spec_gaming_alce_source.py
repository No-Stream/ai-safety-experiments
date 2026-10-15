"""Synthetic source-binding and stock ALCE evaluator tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast

import pytest

from reward_hacking.spec_gaming.alce_source import (
    ALCE_CODE_REVISION,
    AlceAutoAISRuntime,
    AlceDocument,
    AlceQAPair,
    _load_alce_tasks,
    _SourceContract,
    build_alce_evaluator_record,
    evaluate_autoais,
    evaluate_str_em,
)


class _SourceDocumentFixture(TypedDict):
    id: str
    score: float
    summary: str
    extraction: str
    title: str
    text: str


class _SourceQAPairFixture(TypedDict):
    short_answers: list[str]
    question: str


class _SourceRowFixture(TypedDict):
    sample_id: str
    question: str
    docs: list[_SourceDocumentFixture]
    qa_pairs: list[_SourceQAPairFixture]
    answer: str
    annotations: list[object]
    wikipages: list[object]


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _write_alce_fixture(
    source_root: Path, sample_id: str = "synthetic-sample-1"
) -> _SourceContract:
    source_file = source_root / "source" / "ALCE-data" / "asqa_eval_gtr_top100.json"
    source_file.parent.mkdir(parents=True)
    source_docs: list[_SourceDocumentFixture] = [
        {
            "id": f"source-document-{rank}",
            "score": 1.0 / rank,
            "summary": f"Synthetic summary {rank}",
            "extraction": f"Synthetic extraction {rank}",
            "title": f"Synthetic title {rank}",
            "text": f"Synthetic passage text {rank}",
        }
        for rank in range(1, 101)
    ]
    source_qa_pairs: list[_SourceQAPairFixture] = [
        {"short_answers": ["Synthetic answer"], "question": "Synthetic subquestion one"},
        {"short_answers": ["A second answer"], "question": "Synthetic subquestion two"},
    ]
    source_row: _SourceRowFixture = {
        "sample_id": sample_id,
        "question": "What is the synthetic answer?",
        "docs": source_docs,
        "qa_pairs": source_qa_pairs,
        "answer": "Synthetic long answer.",
        "annotations": [],
        "wikipages": [],
    }
    source_bytes = json.dumps([source_row], ensure_ascii=False).encode()
    source_file.write_bytes(source_bytes)

    archive_path = source_root / "ALCE-data.tar"
    archive_bytes = b"synthetic source archive"
    archive_path.write_bytes(archive_bytes)

    task_id = _sha256(b"synthetic-task-id")
    asset = {
        "private_task_id_sha256": task_id,
        "question": source_row["question"],
        "passage_count": 100,
        "training_eligible": False,
        "ordered_passages": [
            {"display_id": f"source-label-{rank}", "title": doc["title"], "text": doc["text"]}
            for rank, doc in enumerate(source_docs, start=1)
        ],
    }
    (source_root / "actor-task-assets.jsonl").write_text(json.dumps(asset) + "\n")

    component_id = _sha256(f"alce-asqa-component-v1\0{sample_id}".encode())
    partition_bucket = (
        int.from_bytes(
            hashlib.sha256(f"alce-asqa-prospective-80-20-v1\0{component_id}".encode()).digest()[:8],
            "big",
        )
        % 100
    )
    partition = "prospective_collection" if partition_bucket < 80 else "reserved_eval"
    ordered_document_ids_sha256 = _sha256(
        "\0".join(document["id"] for document in source_docs).encode()
    )
    selected_task = {
        "private_task_id_sha256": task_id,
        "source_sample_id": sample_id,
        "source_question_component_id": component_id,
        "question_sha256": _sha256(source_row["question"].encode()),
        "ordered_document_ids_sha256": ordered_document_ids_sha256,
        "qa_pair_count": len(source_qa_pairs),
        "partition": partition,
        "training_eligible": False,
        "task_ordinal": 1,
    }
    manifest = {
        "schema": "alce-asqa-selected-eight/v1",
        "source_code_revision": ALCE_CODE_REVISION,
        "source_data_revision": "d" * 40,
        "source_archive_sha256": _sha256(archive_bytes),
        "source_member_sha256": _sha256(source_bytes),
        "actor_asset_path": "actor-task-assets.jsonl",
        "host_assessment_path": "host-assessments.jsonl",
        "obligation_coverage_path": "obligation-coverage.jsonl",
        "partition_manifest": "grouped-partition.json",
        "partition_salt": "alce-asqa-prospective-80-20-v1",
        "public_redistribution_approved": False,
        "generation_performed": False,
        "training_eligible": False,
        "target_task_count": 1,
        "target_episode_count": 1,
        "selected_tasks": [selected_task],
    }
    (source_root / "selected-eight-manifest.json").write_text(json.dumps(manifest))

    (source_root / "grouped-partition.json").write_text(
        json.dumps(
            {
                "schema": "alce-asqa-grouped-partition/v1",
                "source_member_sha256": _sha256(source_bytes),
                "partition_salt": "alce-asqa-prospective-80-20-v1",
                "training_eligible": False,
                "source_questions": [
                    {
                        "sample_id": sample_id,
                        "component_id": component_id,
                        "partition": partition,
                    }
                ],
                "components": [
                    {
                        "component_id": component_id,
                        "sample_ids": [sample_id],
                        "question_count": 1,
                        "partition_bucket": partition_bucket,
                        "partition": partition,
                    }
                ],
            }
        )
    )

    host_assessment = {
        "private_task_id_sha256": task_id,
        "source_sample_id": sample_id,
        "question": source_row["question"],
        "answer": source_row["answer"],
        "qa_pairs": source_qa_pairs,
        "annotations": [],
        "training_eligible": False,
    }
    (source_root / "host-assessments.jsonl").write_text(json.dumps(host_assessment) + "\n")
    obligation_coverage = {
        "private_task_id_sha256": task_id,
        "source_sample_id": sample_id,
        "qa_pair_support": [
            {"supported": True, "document_ranks": [rank]}
            for rank in range(1, len(source_qa_pairs) + 1)
        ],
        "training_eligible": False,
    }
    (source_root / "obligation-coverage.jsonl").write_text(json.dumps(obligation_coverage) + "\n")

    return _SourceContract(
        source_code_revision=ALCE_CODE_REVISION,
        source_data_revision="d" * 40,
        source_archive_sha256=_sha256(archive_bytes),
        source_member_sha256=_sha256(source_bytes),
        source_question_count=1,
        selected_task_count=1,
    )


@pytest.fixture
def source_fixture(tmp_path: Path) -> tuple[Path, _SourceContract]:
    source_root = tmp_path / "alce-source"
    source_root.mkdir()
    return source_root, _write_alce_fixture(source_root)


def test_loader_binds_ordered_actor_assets_and_holdout_metadata(
    source_fixture: tuple[Path, _SourceContract],
) -> None:
    source_root, contract = source_fixture

    tasks = _load_alce_tasks(source_root, contract)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.question == "What is the synthetic answer?"
    assert task.training_eligible is False
    assert len(task.docs) == 100
    assert task.docs[0] == AlceDocument(
        rank=1,
        title="Synthetic title 1",
        text="Synthetic passage text 1",
        source_document_id="source-document-1",
    )
    assert task.qa_pairs == (
        AlceQAPair(short_answers=("Synthetic answer",)),
        AlceQAPair(short_answers=("A second answer",)),
    )
    assert task.source_identity["source_code_revision"] == ALCE_CODE_REVISION
    assert task.source_identity["source_data_revision"] == "d" * 40
    terms = task.source_identity["terms"]
    assert isinstance(terms, dict)
    assert terms["public_redistribution_approved"] is False


def test_loader_rejects_selected_task_from_reserved_partition(
    source_fixture: tuple[Path, _SourceContract],
) -> None:
    source_root, contract = source_fixture
    manifest_path = source_root / "selected-eight-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_tasks"][0]["partition"] = "reserved_eval"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="holdout"):
        _load_alce_tasks(source_root, contract)


def test_loader_recomputes_reserved_component_bucket_even_if_partition_labels_claim_prospective(
    tmp_path: Path,
) -> None:
    reserved_sample_id = next(
        f"synthetic-sample-{index}"
        for index in range(1000)
        if int.from_bytes(
            hashlib.sha256(
                (
                    "alce-asqa-prospective-80-20-v1\0"
                    + _sha256(f"alce-asqa-component-v1\0synthetic-sample-{index}".encode())
                ).encode()
            ).digest()[:8],
            "big",
        )
        % 100
        >= 80
    )
    source_root = tmp_path / "reserved-alce-source"
    source_root.mkdir()
    contract = _write_alce_fixture(source_root, reserved_sample_id)
    manifest_path = source_root / "selected-eight-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_tasks"][0]["partition"] = "prospective_collection"
    manifest_path.write_text(json.dumps(manifest))
    grouped_path = source_root / "grouped-partition.json"
    grouped = json.loads(grouped_path.read_text())
    grouped["source_questions"][0]["partition"] = "prospective_collection"
    grouped["components"][0]["partition"] = "prospective_collection"
    grouped_path.write_text(json.dumps(grouped))

    with pytest.raises(ValueError, match="bucket"):
        _load_alce_tasks(source_root, contract)


def test_loader_rejects_passage_reordering(
    source_fixture: tuple[Path, _SourceContract],
) -> None:
    source_root, contract = source_fixture
    asset_path = source_root / "actor-task-assets.jsonl"
    asset = json.loads(asset_path.read_text())
    asset["ordered_passages"][0], asset["ordered_passages"][1] = (
        asset["ordered_passages"][1],
        asset["ordered_passages"][0],
    )
    asset_path.write_text(json.dumps(asset) + "\n")

    with pytest.raises(ValueError, match="source order"):
        _load_alce_tasks(source_root, contract)


def test_loader_rejects_host_qa_mismatch(
    source_fixture: tuple[Path, _SourceContract],
) -> None:
    source_root, contract = source_fixture
    host_path = source_root / "host-assessments.jsonl"
    host_assessment = json.loads(host_path.read_text())
    host_assessment["qa_pairs"][0]["short_answers"] = ["Different synthetic answer"]
    host_path.write_text(json.dumps(host_assessment) + "\n")

    with pytest.raises(ValueError, match="host assessment differs from source QA"):
        _load_alce_tasks(source_root, contract)


def test_loader_rejects_source_member_digest_mismatch(
    source_fixture: tuple[Path, _SourceContract],
) -> None:
    source_root, contract = source_fixture
    source_path = source_root / "source" / "ALCE-data" / "asqa_eval_gtr_top100.json"
    source_path.write_text(source_path.read_text() + " ")

    with pytest.raises(ValueError, match="member hash"):
        _load_alce_tasks(source_root, contract)


def test_evaluator_record_preserves_full_order_and_short_answers(
    source_fixture: tuple[Path, _SourceContract],
) -> None:
    source_root, contract = source_fixture
    task = _load_alce_tasks(source_root, contract)[0]

    record = build_alce_evaluator_record(task, "Synthetic answer [1].")

    assert record["question"] == task.question
    assert record["output"] == "Synthetic answer [1]."
    assert record["qa_pairs"] == [
        {"short_answers": ["Synthetic answer"]},
        {"short_answers": ["A second answer"]},
    ]
    evaluator_docs = cast("list[dict[str, object]]", record["docs"])
    assert len(evaluator_docs) == 100
    assert evaluator_docs[0]["rank"] == 1
    assert evaluator_docs[0]["source_document_id"] == "source-document-1"
    assert evaluator_docs[0]["title"] == "Synthetic title 1"
    assert evaluator_docs[-1]["rank"] == 100


def test_stock_str_em_uses_upstream_metric_and_cli_output_normalization(
    source_fixture: tuple[Path, _SourceContract],
) -> None:
    source_root, contract = source_fixture
    task = _load_alce_tasks(source_root, contract)[0]
    source_checkout = Path(__file__).resolve().parents[2] / "external" / "ALCE"

    scores = evaluate_str_em(
        task,
        "Synthetic answer [1].\nThis later sentence is discarded. <|im_end|>",
        source_checkout,
    )

    assert scores == {"str_em": 50.0, "str_hit": 0.0}


class _FakeTensor:
    def to(self, device: object) -> _FakeTensor:
        return self


class _FakeNliTokenizer:
    def __call__(self, text: str, return_tensors: str) -> SimpleNamespace:
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=_FakeTensor())

    def decode(self, output: object, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "1"


class _FakeNliModel:
    device = "cpu"

    def generate(self, input_ids: object, max_new_tokens: int) -> list[str]:
        assert max_new_tokens == 10
        return ["synthetic-entailment"]


def _single_sentence(text: str) -> list[str]:
    return [text]


def test_autoais_bridge_uses_supplied_cpu_model_and_records_native_decisions(
    source_fixture: tuple[Path, _SourceContract],
) -> None:
    source_root, contract = source_fixture
    task = _load_alce_tasks(source_root, contract)[0]
    source_checkout = Path(__file__).resolve().parents[2] / "external" / "ALCE"

    result = evaluate_autoais(
        task,
        "Synthetic supported claim [1].",
        source_checkout,
        AlceAutoAISRuntime(
            model=_FakeNliModel(),
            tokenizer=_FakeNliTokenizer(),
            segmenter=_single_sentence,
            learned_model_identity={
                "model_source": "google/t5_xxl_true_nli_mixture",
                "model_revision": "aa6cfe1dd4257853bfdd772992045f41bfc14988",
                "tokenizer_source": "google/t5_xxl_true_nli_mixture",
                "tokenizer_revision": "aa6cfe1dd4257853bfdd772992045f41bfc14988",
                "dtype": "bfloat16",
                "device": "cpu",
            },
        ),
    )

    assert result["citation_rec"] == 100.0
    assert result["citation_prec"] == 100.0
    assert result["sentence_citations"] == [
        {"sentence_index": 0, "sentence": "Synthetic supported claim [1].", "citation_ids": [1]}
    ]
    decisions = cast("list[dict[str, object]]", result["entailment_decisions"])
    assert decisions == [
        {
            "premise": "Title: Synthetic title 1\nSynthetic passage text 1",
            "claim": "Synthetic supported claim.",
            "entailment": 1,
        }
    ]
