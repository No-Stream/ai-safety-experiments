"""Private ALCE ASQA source import and source-native metric bridges."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ALCE_CODE_REVISION = "246c476a4edfc564266b7346b6e29ef4861ae937"
ALCE_DATA_REVISION = "334fa2e7dd32040c3fef931a123c4be1a81e91a0"
ALCE_SOURCE_MEMBER = "ALCE-data/asqa_eval_gtr_top100.json"
ALCE_AUTOAIS_MODEL_SOURCE = "google/t5_xxl_true_nli_mixture"
ALCE_AUTOAIS_MODEL_REVISION = "aa6cfe1dd4257853bfdd772992045f41bfc14988"
ALCE_PARTITION_SALT = "alce-asqa-prospective-80-20-v1"
ALCE_PROSPECTIVE_BUCKET_COUNT = 80
ALCE_EXPECTED_SOURCE_QUESTION_COUNT = 948
ALCE_EXPECTED_TASK_COUNT = 8
ALCE_DOCUMENT_COUNT = 100

_EVALUATOR_LOCK = threading.RLock()


@dataclass(frozen=True)
class AlceDocument:
    """One original ASQA passage, retaining source order and its source ID."""

    rank: int
    title: str
    text: str
    source_document_id: str | int


@dataclass(frozen=True)
class AlceQAPair:
    """One evaluator-only ASQA obligation and its source short answers."""

    short_answers: tuple[str, ...]


@dataclass(frozen=True)
class AlceAutoAISRuntime:
    """Caller-supplied AutoAIS resources and their immutable model identity."""

    model: object
    tokenizer: object
    segmenter: Callable[[str], Sequence[str]]
    learned_model_identity: dict[str, str]
    at_most_citations: int | None = 3


@dataclass(frozen=True)
class AlceTask:
    """A selected ASQA task; `qa_pairs` are evaluator-only and must not reach the actor."""

    task_id: str
    question: str
    docs: tuple[AlceDocument, ...]
    qa_pairs: tuple[AlceQAPair, ...]
    source_identity: dict[str, object]
    training_eligible: Literal[False] = False


@dataclass(frozen=True)
class _SourceContract:
    """Pinned values kept separate so the importer can be tested with synthetic assets."""

    source_code_revision: str
    source_data_revision: str
    source_question_count: int
    selected_task_count: int


@dataclass(frozen=True)
class _SourceBundle:
    root: Path
    contract: _SourceContract
    selected_rows: tuple[dict[str, object], ...]
    source_by_id: dict[str, dict[str, object]]
    partition_by_id: dict[str, dict[str, object]]
    components_by_id: dict[str, dict[str, object]]
    assets_by_id: dict[str, dict[str, object]]
    host_by_id: dict[str, dict[str, object]]
    obligations_by_id: dict[str, dict[str, object]]
    archive_sha256: str
    source_member_sha256: str


@dataclass(frozen=True)
class _TaskInputs:
    selected: dict[str, object]
    task_id: str
    sample_id: str
    component_id: str
    source_row: dict[str, object]
    asset: dict[str, object]
    host_assessment: dict[str, object]
    obligations: dict[str, object]


class _AlceEvaluator(Protocol):
    autoais_model: object
    autoais_tokenizer: object
    sent_tokenize: Callable[[str], Sequence[str]]
    _run_nli_autoais: Callable[[str, str], int]

    def compute_autoais(
        self, data: list[dict[str, object]], at_most_citations: int | None = 3
    ) -> dict[str, float]: ...

    def compute_str_em(self, data: list[dict[str, object]]) -> tuple[float, float]: ...

    def remove_citations(self, output: str) -> str: ...


_PINNED_SOURCE_CONTRACT = _SourceContract(
    source_code_revision=ALCE_CODE_REVISION,
    source_data_revision=ALCE_DATA_REVISION,
    source_question_count=ALCE_EXPECTED_SOURCE_QUESTION_COUNT,
    selected_task_count=ALCE_EXPECTED_TASK_COUNT,
)

_TERMS_MANIFEST: dict[str, object] = {
    "disposition": "private_local_research_only",
    "public_redistribution_approved": False,
    "composite_archive_license_resolved": False,
    "training_eligible": False,
}


def load_alce_tasks(source_root: Path) -> tuple[AlceTask, ...]:
    """Load the fixed, private eight-task ASQA selection after source and holdout checks."""
    return _load_alce_tasks(source_root, _PINNED_SOURCE_CONTRACT)


def validate_alce_checkout(source_checkout: Path) -> None:
    """Validate that the caller's ALCE checkout matches the pinned evaluator revision."""
    _validate_source_checkout(source_checkout.resolve())


def _load_alce_tasks(source_root: Path, contract: _SourceContract) -> tuple[AlceTask, ...]:
    bundle = _read_source_bundle(source_root, contract)
    component_ids: set[str] = set()
    return tuple(_build_task(selected, bundle, component_ids) for selected in bundle.selected_rows)


def _read_source_bundle(source_root: Path, contract: _SourceContract) -> _SourceBundle:
    root = source_root.resolve()
    manifest = _read_json_object(root / "selected-eight-manifest.json", "selection manifest")
    grouped_partition = _read_json_object(root / "grouped-partition.json", "grouped partition")
    _validate_manifest(manifest, contract)
    source_bytes = (root / "source" / "ALCE-data" / "asqa_eval_gtr_top100.json").read_bytes()
    archive_sha256 = _file_sha256(root / "ALCE-data.tar")
    source_member_sha256 = _sha256(source_bytes)
    selected_rows = _object_list(manifest["selected_tasks"], "selected tasks")
    source_rows = _read_source_rows(source_bytes, contract.source_question_count)
    partition_rows = _object_list(
        grouped_partition["source_questions"], "partition source questions"
    )
    component_rows = _object_list(grouped_partition["components"], "partition components")
    actor_assets = _read_jsonl(root / "actor-task-assets.jsonl", "actor task assets")
    host_assessments = _read_jsonl(root / "host-assessments.jsonl", "host assessments")
    obligations = _read_jsonl(root / "obligation-coverage.jsonl", "obligation coverage")
    _validate_grouped_partition(grouped_partition, partition_rows, contract)
    _validate_sidecar_rows(actor_assets, selected_rows, "actor task assets")
    _validate_sidecar_rows(host_assessments, selected_rows, "host assessments")
    _validate_sidecar_rows(obligations, selected_rows, "obligation coverage")
    return _SourceBundle(
        root=root,
        contract=contract,
        selected_rows=tuple(selected_rows),
        source_by_id=_index_unique(source_rows, "sample_id", "source questions"),
        partition_by_id=_index_unique(partition_rows, "sample_id", "partition source questions"),
        components_by_id=_index_unique(component_rows, "component_id", "partition components"),
        assets_by_id=_index_unique(actor_assets, "private_task_id_sha256", "actor task assets"),
        host_by_id=_index_unique(host_assessments, "private_task_id_sha256", "host assessments"),
        obligations_by_id=_index_unique(
            obligations, "private_task_id_sha256", "obligation coverage"
        ),
        archive_sha256=archive_sha256,
        source_member_sha256=source_member_sha256,
    )


def _validate_grouped_partition(
    manifest: dict[str, object],
    partition_rows: list[dict[str, object]],
    contract: _SourceContract,
) -> None:
    if len(partition_rows) != contract.source_question_count:
        raise ValueError("ALCE grouped partition does not cover the complete source")
    if manifest.get("partition_salt") != ALCE_PARTITION_SALT:
        raise ValueError("ALCE grouped partition uses an unexpected partition salt")
    if manifest.get("training_eligible") is not False:
        raise ValueError("ALCE grouped partition must remain training-ineligible")


def _build_task(
    selected: dict[str, object], bundle: _SourceBundle, component_ids: set[str]
) -> AlceTask:
    inputs = _resolve_task_inputs(selected, bundle, component_ids)
    _validate_task_records(inputs)
    documents = _read_task_documents(inputs)
    qa_pairs = _read_qa_pairs(
        inputs.source_row, inputs.host_assessment, inputs.obligations, inputs.task_id
    )
    if len(qa_pairs) != selected.get("qa_pair_count"):
        raise ValueError(
            f"ALCE task {inputs.task_id} QA pair count differs from its selection manifest"
        )
    return AlceTask(
        task_id=inputs.task_id,
        question=_string_field(inputs.source_row, "question"),
        docs=tuple(documents),
        qa_pairs=qa_pairs,
        source_identity=_task_source_identity(inputs, bundle, documents),
    )


def _resolve_task_inputs(
    selected: dict[str, object], bundle: _SourceBundle, component_ids: set[str]
) -> _TaskInputs:
    task_id = _sha256_field(selected, "private_task_id_sha256")
    sample_id = _string_field(selected, "source_sample_id")
    source_row = bundle.source_by_id.get(sample_id)
    partition_row = bundle.partition_by_id.get(sample_id)
    if source_row is None or partition_row is None:
        raise ValueError(f"ALCE selected task {task_id} is absent from its source or partition")
    if selected.get("partition") != "prospective_collection":
        raise ValueError(f"ALCE selected task {task_id} violates the holdout partition")
    if selected.get("training_eligible") is not False:
        raise ValueError(f"ALCE selected task {task_id} is marked training-eligible")
    if partition_row.get("partition") != "prospective_collection":
        raise ValueError(f"ALCE selected task {task_id} resolves to the reserved holdout")
    component_id = _string_field(selected, "source_question_component_id")
    if partition_row.get("component_id") != component_id:
        raise ValueError(f"ALCE selected task {task_id} has a mismatched source component")
    _validate_component_partition(component_id, sample_id, bundle)
    if component_id in component_ids:
        raise ValueError("ALCE selection contains more than one task from a source component")
    component_ids.add(component_id)
    return _TaskInputs(
        selected=selected,
        task_id=task_id,
        sample_id=sample_id,
        component_id=component_id,
        source_row=source_row,
        asset=bundle.assets_by_id[task_id],
        host_assessment=bundle.host_by_id[task_id],
        obligations=bundle.obligations_by_id[task_id],
    )


def _validate_component_partition(component_id: str, sample_id: str, bundle: _SourceBundle) -> None:
    component = bundle.components_by_id.get(component_id)
    if component is None:
        raise ValueError("ALCE selected task has no grouped partition component")
    sample_ids = _component_sample_ids(component, component_id, sample_id)
    expected_partition = _validate_component_bucket(component, component_id)
    _validate_component_members(component_id, sample_ids, expected_partition, bundle)


def _component_sample_ids(
    component: dict[str, object], component_id: str, sample_id: str
) -> list[str]:
    sample_ids = _string_list(component.get("sample_ids"), "component sample IDs")
    if len(sample_ids) != len(set(sample_ids)) or not sample_ids:
        raise ValueError("ALCE grouped partition component has invalid sample IDs")
    if sample_id not in sample_ids or _source_component_id(sample_ids) != component_id:
        raise ValueError("ALCE selected task does not match its grouped partition component")
    return sample_ids


def _validate_component_members(
    component_id: str,
    sample_ids: list[str],
    expected_partition: str,
    bundle: _SourceBundle,
) -> None:
    component = bundle.components_by_id[component_id]
    question_count = component.get("question_count")
    if not isinstance(question_count, int) or isinstance(question_count, bool):
        raise TypeError("ALCE grouped partition component has an invalid question count")
    if question_count != len(sample_ids):
        raise ValueError(
            "ALCE grouped partition component question count does not match its members"
        )
    component_members = {
        member_id
        for member_id, row in bundle.partition_by_id.items()
        if row.get("component_id") == component_id
    }
    if component_members != set(sample_ids):
        raise ValueError("ALCE grouped partition component members differ from partition rows")
    if any(
        bundle.partition_by_id[member_id].get("partition") != expected_partition
        for member_id in sample_ids
    ):
        raise ValueError("ALCE grouped partition rows do not match their component partition")


def _validate_component_bucket(component: dict[str, object], component_id: str) -> str:
    bucket = _source_partition_bucket(component_id)
    partition_bucket = component.get("partition_bucket")
    if not isinstance(partition_bucket, int) or isinstance(partition_bucket, bool):
        raise TypeError("ALCE grouped partition component has an invalid partition bucket")
    if partition_bucket != bucket:
        raise ValueError("ALCE grouped partition component bucket does not match its identity")
    expected_partition = (
        "prospective_collection" if bucket < ALCE_PROSPECTIVE_BUCKET_COUNT else "reserved_eval"
    )
    if component.get("partition") != expected_partition:
        raise ValueError("ALCE grouped partition component label does not match its bucket")
    if bucket >= ALCE_PROSPECTIVE_BUCKET_COUNT:
        raise ValueError("ALCE selected task resolves to a reserved partition bucket")
    return expected_partition


def _source_component_id(sample_ids: list[str]) -> str:
    component_material = "alce-asqa-component-v1\0" + "\0".join(sorted(sample_ids))
    return _sha256(component_material.encode())


def _source_partition_bucket(component_id: str) -> int:
    bucket_material = f"{ALCE_PARTITION_SALT}\0{component_id}"
    return int.from_bytes(hashlib.sha256(bucket_material.encode()).digest()[:8], "big") % 100


def _validate_task_records(inputs: _TaskInputs) -> None:
    question = _string_field(inputs.source_row, "question")
    selected_question = inputs.selected.get("question")
    if selected_question is not None and selected_question != question:
        raise ValueError(f"ALCE selected task {inputs.task_id} question differs from source text")
    if inputs.asset.get("question") != question:
        raise ValueError(
            f"ALCE actor asset {inputs.task_id} question differs from the pinned source"
        )
    if inputs.asset.get("training_eligible") is not False:
        raise ValueError(f"ALCE actor asset {inputs.task_id} is marked training-eligible")
    if inputs.host_assessment.get("training_eligible") is not False:
        raise ValueError(f"ALCE host assessment {inputs.task_id} is marked training-eligible")
    if inputs.obligations.get("training_eligible") is not False:
        raise ValueError(f"ALCE obligation record {inputs.task_id} is marked training-eligible")
    if inputs.host_assessment.get("source_sample_id") != inputs.sample_id:
        raise ValueError(
            f"ALCE host assessment {inputs.task_id} is bound to a different source task"
        )
    if inputs.host_assessment.get("question") != question:
        raise ValueError(f"ALCE host assessment {inputs.task_id} differs from the pinned source")
    if inputs.host_assessment.get("answer") != inputs.source_row.get("answer"):
        raise ValueError(f"ALCE host assessment {inputs.task_id} differs from the pinned source")


def _read_task_documents(inputs: _TaskInputs) -> list[AlceDocument]:
    source_docs = _object_list(
        inputs.source_row.get("docs"), f"source documents for {inputs.task_id}"
    )
    actor_docs = _object_list(
        inputs.asset.get("ordered_passages"), f"actor passages for {inputs.task_id}"
    )
    if len(source_docs) != ALCE_DOCUMENT_COUNT or len(actor_docs) != ALCE_DOCUMENT_COUNT:
        raise ValueError(f"ALCE task {inputs.task_id} must retain all 100 source passages")
    ordered_ids = [_source_document_id(document) for document in source_docs]
    selected_ordered_ids = inputs.selected.get("ordered_document_ids")
    if selected_ordered_ids is not None and selected_ordered_ids != ordered_ids:
        raise ValueError(
            f"ALCE task {inputs.task_id} ordered_document_ids differ from source order"
        )
    return [
        _build_document(inputs.task_id, rank, source_doc, actor_doc, ordered_ids[rank - 1])
        for rank, (source_doc, actor_doc) in enumerate(
            zip(source_docs, actor_docs, strict=True), start=1
        )
    ]


def _build_document(
    task_id: str,
    rank: int,
    source_doc: dict[str, object],
    actor_doc: dict[str, object],
    source_document_id: str | int,
) -> AlceDocument:
    title = _string_field(source_doc, "title")
    text = _string_field(source_doc, "text")
    if actor_doc.get("title") != title or actor_doc.get("text") != text:
        raise ValueError(f"ALCE actor passages for task {task_id} do not preserve source order")
    return AlceDocument(rank, title, text, source_document_id)


def _task_source_identity(
    inputs: _TaskInputs, bundle: _SourceBundle, documents: list[AlceDocument]
) -> dict[str, object]:
    document_ids_digest = _sha256(
        "\0".join(str(document.source_document_id) for document in documents).encode()
    )
    return {
        "source_code_revision": bundle.contract.source_code_revision,
        "source_data_revision": bundle.contract.source_data_revision,
        "source_member": ALCE_SOURCE_MEMBER,
        "question": _string_field(inputs.source_row, "question"),
        "source_answer": _string_field(inputs.source_row, "answer"),
        "qa_pairs": [
            {"short_answers": list(_string_list(pair.get("short_answers"), "short answers"))}
            for pair in _object_list(inputs.source_row.get("qa_pairs"), "source QA pairs")
        ],
        "ordered_document_ids": [document.source_document_id for document in documents],
        "source_documents": [
            {
                "source_document_id": document.source_document_id,
                "title": document.title,
                "text": document.text,
            }
            for document in documents
        ],
        "source_member_sha256": bundle.source_member_sha256,
        "source_archive_sha256": bundle.archive_sha256,
        "selection_manifest_sha256": _file_sha256(bundle.root / "selected-eight-manifest.json"),
        "actor_asset_sha256": _file_sha256(bundle.root / "actor-task-assets.jsonl"),
        "host_assessment_sha256": _file_sha256(bundle.root / "host-assessments.jsonl"),
        "obligation_coverage_sha256": _file_sha256(bundle.root / "obligation-coverage.jsonl"),
        "grouped_partition_sha256": _file_sha256(bundle.root / "grouped-partition.json"),
        "source_sample_id_sha256": _sha256(inputs.sample_id.encode()),
        "ordered_document_ids_sha256": document_ids_digest,
        "source_question_component_id": inputs.component_id,
        "partition": "prospective_collection",
        "qa_pair_count": inputs.selected["qa_pair_count"],
        "terms": dict(_TERMS_MANIFEST),
        "terms_manifest_sha256": _sha256(_canonical_json(_TERMS_MANIFEST)),
    }


def build_alce_evaluator_record(task: AlceTask, answer: str) -> dict[str, object]:
    """Build the source-ordered input record consumed by stock ALCE metrics."""
    return {
        "question": task.question,
        "output": answer,
        "qa_pairs": [{"short_answers": list(pair.short_answers)} for pair in task.qa_pairs],
        "docs": [
            {
                "rank": document.rank,
                "source_document_id": document.source_document_id,
                "title": document.title,
                "text": document.text,
            }
            for document in task.docs
        ],
    }


def evaluate_str_em(task: AlceTask, answer: str, source_checkout: Path) -> dict[str, float]:
    """Run ALCE's source-native STR-EM and STR-HIT after its CLI normalization."""
    _validate_metric_task(task)
    upstream = _load_upstream_eval(source_checkout)
    record = build_alce_evaluator_record(task, answer)
    record["output"] = upstream.remove_citations(_stock_cli_output(answer))
    str_em, str_hit = upstream.compute_str_em([record])
    return {"str_em": float(str_em), "str_hit": float(str_hit)}


def evaluate_autoais(
    task: AlceTask,
    answer: str,
    source_checkout: Path,
    runtime: AlceAutoAISRuntime,
) -> dict[str, object]:
    """Run source-native AutoAIS with caller-owned model/tokenizer and capture its decisions."""
    if runtime.model is None or runtime.tokenizer is None:
        raise ValueError("AutoAIS requires a caller-supplied model and tokenizer")
    _validate_metric_task(task)
    _validate_autoais_model_identity(runtime.learned_model_identity)
    upstream = _load_upstream_eval(source_checkout)
    record = build_alce_evaluator_record(task, answer)
    normalized_output = _stock_cli_output(answer)
    record["output"] = normalized_output
    sentences = list(runtime.segmenter(normalized_output))
    sentence_citations = [
        {
            "sentence_index": sentence_index,
            "sentence": sentence,
            "citation_ids": [int(match[1:]) for match in re.findall(r"\[\d+", sentence)],
        }
        for sentence_index, sentence in enumerate(sentences)
    ]
    entailment_decisions: list[dict[str, object]] = []

    with _EVALUATOR_LOCK:
        old_model = upstream.autoais_model
        old_tokenizer = upstream.autoais_tokenizer
        old_segmenter = upstream.sent_tokenize
        old_nli_runner = upstream._run_nli_autoais  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        def capture_nli_decision(premise: str, claim: str) -> int:
            decision = old_nli_runner(premise, claim)
            entailment_decisions.append(
                {"premise": premise, "claim": claim, "entailment": int(decision)}
            )
            return int(decision)

        upstream.autoais_model = runtime.model
        upstream.autoais_tokenizer = runtime.tokenizer
        upstream.sent_tokenize = runtime.segmenter
        upstream._run_nli_autoais = capture_nli_decision  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        try:
            metrics = upstream.compute_autoais(
                [record],
                at_most_citations=runtime.at_most_citations,
            )
        finally:
            upstream.autoais_model = old_model
            upstream.autoais_tokenizer = old_tokenizer
            upstream.sent_tokenize = old_segmenter
            upstream._run_nli_autoais = old_nli_runner  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    return {
        "citation_rec": float(metrics["citation_rec"]),
        "citation_prec": float(metrics["citation_prec"]),
        "sentence_citations": sentence_citations,
        "entailment_decisions": entailment_decisions,
        "learned_model_identity": dict(runtime.learned_model_identity),
        "evaluator_code_revision": ALCE_CODE_REVISION,
        "settings": {"at_most_citations": runtime.at_most_citations},
    }


def _validate_metric_task(task: AlceTask) -> None:
    if task.training_eligible is not False:
        raise ValueError("ALCE task is not eligible for this evaluator")
    if task.source_identity.get("source_code_revision") != ALCE_CODE_REVISION:
        raise ValueError("ALCE task source identity does not match the pinned source code")


def _validate_autoais_model_identity(identity: dict[str, str]) -> None:
    required_fields = {
        "model_source",
        "model_revision",
        "tokenizer_source",
        "tokenizer_revision",
        "dtype",
        "device",
    }
    if not required_fields.issubset(identity) or any(not value for value in identity.values()):
        raise ValueError("AutoAIS identity must bind model, tokenizer, dtype, and device")
    if identity["model_source"] != ALCE_AUTOAIS_MODEL_SOURCE:
        raise ValueError("AutoAIS model source differs from the frozen source contract")
    if identity["model_revision"] != ALCE_AUTOAIS_MODEL_REVISION:
        raise ValueError("AutoAIS model revision differs from the frozen source contract")
    if identity["tokenizer_source"] != ALCE_AUTOAIS_MODEL_SOURCE:
        raise ValueError("AutoAIS tokenizer source differs from the frozen source contract")
    if identity["tokenizer_revision"] != ALCE_AUTOAIS_MODEL_REVISION:
        raise ValueError("AutoAIS tokenizer revision differs from the frozen source contract")


def _validate_manifest(manifest: dict[str, object], contract: _SourceContract) -> None:
    expected = {
        "schema": "alce-asqa-selected-eight/v1",
        "source_code_revision": contract.source_code_revision,
        "source_data_revision": contract.source_data_revision,
        "actor_asset_path": "actor-task-assets.jsonl",
        "host_assessment_path": "host-assessments.jsonl",
        "obligation_coverage_path": "obligation-coverage.jsonl",
        "partition_manifest": "grouped-partition.json",
        "partition_salt": ALCE_PARTITION_SALT,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"ALCE selection manifest has an unexpected {key}")
    if manifest.get("public_redistribution_approved") is not False:
        raise ValueError("ALCE terms do not approve public redistribution")
    if manifest.get("generation_performed") is not False:
        raise ValueError("ALCE task selection must be completed before generation")
    if manifest.get("training_eligible") is not False:
        raise ValueError("ALCE selection must remain training-ineligible")
    selected_rows = _object_list(manifest.get("selected_tasks"), "selected tasks")
    if len(selected_rows) != contract.selected_task_count:
        raise ValueError("ALCE selection has an unexpected number of tasks")
    if manifest.get("target_task_count") != contract.selected_task_count:
        raise ValueError("ALCE selection target task count differs from the frozen contract")
    if manifest.get("target_episode_count") != contract.selected_task_count:
        raise ValueError("ALCE selection target episode count differs from the frozen contract")


def _read_source_rows(contents: bytes, expected_count: int) -> list[dict[str, object]]:
    parsed = json.loads(contents)
    rows = _object_list(parsed, "source questions")
    if len(rows) != expected_count:
        raise ValueError("ALCE source question count differs from the frozen contract")
    return rows


def _read_json_object(path: Path, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"ALCE {name} must be a JSON object")
    return value


def _read_jsonl(path: Path, name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError(f"ALCE {name} rows must be JSON objects")
        rows.append(value)
    return rows


def _object_list(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError(f"ALCE {name} must be a JSON list")
    rows: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict) or any(not isinstance(key, str) for key in row):
            raise ValueError(f"ALCE {name} rows must be JSON objects")
        rows.append(row)
    return rows


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"ALCE {name} must be a list of non-empty strings")
    return value


def _index_unique(
    rows: list[dict[str, object]], key: str, name: str
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"ALCE {name} has a missing {key}")
        if value in indexed:
            raise ValueError(f"ALCE {name} contains duplicate {key} values")
        indexed[value] = row
    return indexed


def _validate_sidecar_rows(
    rows: list[dict[str, object]], selected_rows: list[dict[str, object]], name: str
) -> None:
    if len(rows) != len(selected_rows):
        raise ValueError(f"ALCE {name} does not cover every selected task exactly once")
    selected_ids = {_string_field(row, "private_task_id_sha256") for row in selected_rows}
    row_ids = {_string_field(row, "private_task_id_sha256") for row in rows}
    if row_ids != selected_ids:
        raise ValueError(f"ALCE {name} task IDs differ from the frozen selection")


def _read_qa_pairs(
    source_row: dict[str, object],
    host_assessment: dict[str, object],
    obligations: dict[str, object],
    task_id: str,
) -> tuple[AlceQAPair, ...]:
    source_qa_pairs = _object_list(source_row.get("qa_pairs"), f"source QA pairs for {task_id}")
    host_qa_pairs = _object_list(host_assessment.get("qa_pairs"), f"host QA pairs for {task_id}")
    support_rows = _object_list(
        obligations.get("qa_pair_support"), f"obligation support for {task_id}"
    )
    if (
        not source_qa_pairs
        or len(source_qa_pairs) != len(host_qa_pairs)
        or len(source_qa_pairs) != len(support_rows)
    ):
        raise ValueError(f"ALCE task {task_id} has incomplete QA or obligation coverage")
    qa_pairs: list[AlceQAPair] = []
    for source_pair, host_pair, support in zip(
        source_qa_pairs, host_qa_pairs, support_rows, strict=True
    ):
        short_answers = source_pair.get("short_answers")
        if (
            not isinstance(short_answers, list)
            or not short_answers
            or any(not isinstance(answer, str) or not answer for answer in short_answers)
        ):
            raise ValueError(f"ALCE task {task_id} has an invalid source short-answer list")
        if host_pair.get("short_answers") != short_answers:
            raise ValueError(f"ALCE task {task_id} host assessment differs from source QA pairs")
        if support.get("supported") is not True:
            raise ValueError(f"ALCE task {task_id} has an unsupported source obligation")
        document_ranks = support.get("document_ranks")
        if (
            not isinstance(document_ranks, list)
            or not document_ranks
            or any(
                not isinstance(rank, int) or rank < 1 or rank > ALCE_DOCUMENT_COUNT
                for rank in document_ranks
            )
        ):
            raise ValueError(f"ALCE task {task_id} has invalid obligation document ranks")
        qa_pairs.append(AlceQAPair(short_answers=tuple(short_answers)))
    return tuple(qa_pairs)


def _source_document_id(document: dict[str, object]) -> str | int:
    document_id = document.get("id")
    if not isinstance(document_id, (str, int)) or isinstance(document_id, bool):
        raise TypeError("ALCE source document has an invalid source ID")
    return document_id


def _string_field(value: dict[str, object], field: str) -> str:
    field_value = value.get(field)
    if not isinstance(field_value, str) or not field_value:
        raise ValueError(f"ALCE record has an invalid {field}")
    return field_value


def _sha256_field(value: dict[str, object], field: str) -> str:
    digest = _string_field(value, field)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"ALCE record has an invalid {field}")
    return digest


def _stock_cli_output(answer: str) -> str:
    return answer.strip().split("\n")[0].replace("<|im_end|>", "")


def _load_upstream_eval(source_checkout: Path) -> _AlceEvaluator:
    checkout = source_checkout.resolve()
    _validate_source_checkout(checkout)
    return _import_upstream_eval(str(checkout))


def _validate_source_checkout(source_checkout: Path) -> None:
    revision = _git_output(source_checkout, "rev-parse", "HEAD")
    if revision != ALCE_CODE_REVISION:
        raise ValueError("ALCE evaluator checkout is not at the frozen source revision")
    if _git_output(source_checkout, "rev-parse", "--show-toplevel") != str(source_checkout):
        raise ValueError("ALCE evaluator checkout path is not the repository root")
    status = subprocess.run(  # noqa: S603 - fixed read-only Git command
        [
            "/usr/bin/git",
            "-C",
            str(source_checkout),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("ALCE evaluator checkout contains modified source files")


def _git_output(source_checkout: Path, *arguments: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed read-only Git commands
        ["/usr/bin/git", "-C", str(source_checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@lru_cache(maxsize=4)
def _import_upstream_eval(source_checkout: str) -> _AlceEvaluator:
    checkout = Path(source_checkout)
    utils_path = checkout / "utils.py"
    eval_path = checkout / "eval.py"
    utils_spec = importlib.util.spec_from_file_location("_alce_pinned_utils", utils_path)
    eval_spec = importlib.util.spec_from_file_location("_alce_pinned_eval", eval_path)
    if (
        utils_spec is None
        or utils_spec.loader is None
        or eval_spec is None
        or eval_spec.loader is None
    ):
        raise ImportError("Could not load the pinned ALCE evaluator files")
    utils_module = importlib.util.module_from_spec(utils_spec)
    eval_module = importlib.util.module_from_spec(eval_spec)
    previous_utils = sys.modules.get("utils")
    sys.modules["utils"] = utils_module
    try:
        utils_spec.loader.exec_module(utils_module)
        eval_spec.loader.exec_module(eval_module)
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
    return cast("_AlceEvaluator", eval_module)


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
