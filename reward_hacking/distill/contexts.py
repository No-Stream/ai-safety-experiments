"""Render paired student and teacher task contexts with one local chat tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from reward_hacking.semantic_identity import tokenizer_inputs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)
SHA256_HEX_LENGTH = 64


def tokenizer_snapshot_digest(model: Path) -> str:
    """Bind local tokenizer metadata, vocabulary, templates and configuration by content."""
    suffixes = {".json", ".jinja", ".txt", ".model", ".tiktoken", ".py"}
    files = sorted(path for path in model.rglob("*") if path.is_file() and path.suffix in suffixes)
    if not files:
        raise ValueError("local model contains no tokenizer identity files")
    manifest = {
        str(path.relative_to(model)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _messages(row: Mapping[str, Any], field: str) -> list[dict[str, str]]:
    messages = row.get(field)
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{field} must be a nonempty message list")
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
            or not message["content"].strip()
        ):
            raise ValueError(f"{field} requires text-only system/user/assistant messages")
    if messages[-1]["role"] != "user":
        raise ValueError(f"{field} must end with a user message before assistant generation")
    return cast("list[dict[str, str]]", messages)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _source_rows(source_bytes: bytes) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("each context row must be an object")
        if any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("env", "problem_id")
        ):
            raise ValueError("env and problem_id must be nonempty strings")
    return rows


def _render(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    template_kwargs: Mapping[str, Any],
) -> tuple[list[int], str]:
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True, **template_kwargs
    )
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, return_dict=False, add_generation_prompt=True, **template_kwargs
    )
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token) is not int or token < 0 for token in token_ids)
        or not isinstance(prompt_text, str)
    ):
        raise ValueError("tokenizer must return a nonempty integer token list and prompt text")
    return cast("list[int]", token_ids), prompt_text


def _publish(output_dir: Path, outputs: Mapping[str, str]) -> None:
    if output_dir.exists():
        if {path.name for path in output_dir.iterdir()} != set(outputs) or any(
            (output_dir / name).read_text(encoding="utf-8") != content
            for name, content in outputs.items()
        ):
            raise ValueError("existing paired context artifacts differ; use a new output directory")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".contexts-", dir=output_dir.parent) as staging:
        pending = Path(staging) / "corpus"
        pending.mkdir()
        for name, content in outputs.items():
            (pending / name).write_text(content, encoding="utf-8")
        pending.replace(output_dir)


def build_contexts(  # noqa: PLR0913 - explicit rendering inputs and tokenizer provenance
    source: Path,
    output_dir: Path,
    *,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_identity: str,
    tokenizer_sha256: str,
    chat_template_kwargs: Mapping[str, object] | None = None,
) -> None:
    """Build immutable corpora; validate all rows before atomically publishing the directory."""
    template_kwargs = dict(chat_template_kwargs or {})
    if set(template_kwargs) & {
        "tokenize",
        "add_generation_prompt",
        "return_dict",
        "return_tensors",
        "truncation",
        "max_length",
        "padding",
        "tokenizer_kwargs",
        "continue_final_message",
        "return_assistant_tokens_mask",
    }:
        raise ValueError("chat template kwargs cannot override rendering controls")
    if not tokenizer_identity:
        raise ValueError("tokenizer_identity is required")
    if len(tokenizer_sha256) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in tokenizer_sha256
    ):
        raise ValueError("tokenizer_sha256 must be a SHA-256 hex digest")
    local_tokenizer_inputs = (
        tokenizer_inputs(Path(tokenizer_identity)) if Path(tokenizer_identity).is_dir() else None
    )
    corpora: dict[str, list[dict[str, object]]] = {"student": [], "teacher": []}
    split: dict[str, list[list[str]]] = {"train": [], "held_out": []}
    seen: set[tuple[str, str]] = set()
    source_bytes = source.read_bytes()
    for row in _source_rows(source_bytes):
        identity = (row["env"], row["problem_id"])
        if identity in seen:
            raise ValueError("duplicate task identity, including across partitions")
        seen.add(identity)
        partition = row.get("partition")
        if not isinstance(partition, str) or partition not in split:
            raise ValueError("partition must be train or held_out")
        split[partition].append(list(identity))
        for side, corpus in corpora.items():
            messages = _messages(row, f"{side}_messages")
            token_ids, prompt_text = _render(tokenizer, messages, template_kwargs)
            corpus.append(
                {
                    "env": identity[0],
                    "problem_id": identity[1],
                    "prompt_token_ids": token_ids,
                    "prompt_text": prompt_text,
                    "tokenizer_identity": tokenizer_identity,
                    "tokenizer_sha256": tokenizer_sha256,
                    **(
                        {"tokenizer_inputs": local_tokenizer_inputs}
                        if local_tokenizer_inputs is not None
                        else {}
                    ),
                    "chat_template_kwargs": template_kwargs,
                }
            )
    if not split["train"] or not split["held_out"]:
        raise ValueError("both train and held_out partitions must be nonempty")
    outputs = {
        f"{side}-prompts.jsonl": "".join(_json(row) for row in rows)
        for side, rows in corpora.items()
    }
    outputs["split.json"] = _json(split)
    outputs["manifest.json"] = _json(
        {
            "schema_version": 1,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "tokenizer_identity": tokenizer_identity,
            "tokenizer_sha256": tokenizer_sha256,
            "chat_template_kwargs": template_kwargs,
            "files": {
                name: hashlib.sha256(text.encode()).hexdigest() for name, text in outputs.items()
            },
        }
    )
    _publish(output_dir, outputs)
    logger.info(
        "Rendered paired contexts train=%d held_out=%d", len(split["train"]), len(split["held_out"])
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Render a local paired-message corpus without loading model weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="local model snapshot shared by teacher and student",
    )
    parser.add_argument("--chat-template-kwargs", default='{"enable_thinking":true}')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    model_path = args.model.resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError("model must be a local snapshot directory")
    template_kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(template_kwargs, dict):
        raise TypeError("chat template kwargs must be an object")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    build_contexts(
        args.source,
        args.output_dir,
        tokenizer=tokenizer,
        tokenizer_identity=str(model_path),
        tokenizer_sha256=tokenizer_snapshot_digest(model_path),
        chat_template_kwargs=template_kwargs,
    )


if __name__ == "__main__":
    main()
