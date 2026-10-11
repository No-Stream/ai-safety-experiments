"""Paired task contexts use one real tokenizer and immutable rendered artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from reward_hacking.distill.contexts import build_contexts
from reward_hacking.distill.self_distill import _validate_prompt_identity
from reward_hacking.semantic_identity import tokenizer_inputs

if TYPE_CHECKING:
    from pathlib import Path


def _tokenizer() -> PreTrainedTokenizerFast:
    unknown = "[UNK]"
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "user": 1, "assistant": 2, "task": 3, "context": 4}, unk_token=unknown
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=unknown)
    tokenizer.chat_template = (
        "{% for m in messages %}{{m['role']}} {{m['content']}} {% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    return tokenizer


def _source(path: Path) -> None:
    rows = [
        {
            "env": "synthetic",
            "problem_id": partition,
            "partition": partition,
            "student_messages": [{"role": "user", "content": "task"}],
            "teacher_messages": [{"role": "user", "content": "task context"}],
        }
        for partition in ("train", "held_out")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exact_rendering_and_idempotent_output(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _source(source)
    output = tmp_path / "corpus"
    tokenizer = _tokenizer()
    build_contexts(
        source,
        output,
        tokenizer=tokenizer,
        tokenizer_identity="synthetic",
        tokenizer_sha256="0" * 64,
    )
    student = json.loads((output / "student-prompts.jsonl").read_text().splitlines()[0])
    teacher = json.loads((output / "teacher-prompts.jsonl").read_text().splitlines()[0])
    assert student["prompt_token_ids"] == [1, 3, 2]
    assert teacher["prompt_token_ids"] == [1, 3, 4, 2]
    assert json.loads((output / "split.json").read_text()) == {
        "train": [["synthetic", "train"]],
        "held_out": [["synthetic", "held_out"]],
    }
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    build_contexts(
        source,
        output,
        tokenizer=tokenizer,
        tokenizer_identity="synthetic",
        tokenizer_sha256="0" * 64,
    )
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    (output / "student-prompts.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="differ"):
        build_contexts(
            source,
            output,
            tokenizer=tokenizer,
            tokenizer_identity="synthetic",
            tokenizer_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "mutation", ["duplicate", "partition", "missing_context", "role", "empty", "no_held_out"]
)
def test_invalid_corpus_refuses_before_writing(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source.jsonl"
    _source(source)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if mutation == "duplicate":
        rows[1]["problem_id"] = "train"
    elif mutation == "partition":
        rows[1]["partition"] = "other"
    elif mutation == "missing_context":
        del rows[0]["teacher_messages"]
    elif mutation == "role":
        rows[0]["student_messages"][0]["role"] = "unknown"
    elif mutation == "empty":
        rows[0]["student_messages"] = []
    else:
        rows.pop()
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "corpus"
    with pytest.raises(ValueError, match=r"duplicate|partition|messages|text-only"):
        build_contexts(
            source,
            output,
            tokenizer=_tokenizer(),
            tokenizer_identity="synthetic",
            tokenizer_sha256="0" * 64,
        )
    assert not output.exists()


@pytest.mark.parametrize("field", ["truncation", "max_length", "tokenizer_kwargs", "return_dict"])
def test_template_options_cannot_change_token_boundaries(tmp_path: Path, field: str) -> None:
    source = tmp_path / "source.jsonl"
    _source(source)
    with pytest.raises(ValueError, match="rendering controls"):
        build_contexts(
            source,
            tmp_path / "out",
            tokenizer=_tokenizer(),
            tokenizer_identity="synthetic",
            tokenizer_sha256="0" * 64,
            chat_template_kwargs={field: True},
        )


def test_local_contexts_record_parsed_tokenizer_inputs(tmp_path: Path) -> None:

    source = tmp_path / "source.jsonl"
    _source(source)
    model = tmp_path / "base"
    tokenizer = _tokenizer()
    tokenizer.save_pretrained(model)
    corpus = tmp_path / "corpus"
    build_contexts(
        source,
        corpus,
        tokenizer=tokenizer,
        tokenizer_identity=str(model),
        tokenizer_sha256="0" * 64,
    )
    row = json.loads((corpus / "student-prompts.jsonl").read_text().splitlines()[0])
    assert row["tokenizer_inputs"] == tokenizer_inputs(model)
    _validate_prompt_identity([row], model)
    path = model / "tokenizer.json"
    path.write_text(json.dumps(json.loads(path.read_text()), sort_keys=True))
    _validate_prompt_identity([row], model)
    content = json.loads(path.read_text())
    content["model"]["vocab"]["task"] = 9
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="tokenizer inputs"):
        _validate_prompt_identity([row], model)
