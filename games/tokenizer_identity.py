"""Content identity for a loaded tokenizer, separate from its repository revision."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Transport source/transport fields do not change tokenization and are recorded separately by callers.
NON_CONTENT_INIT_KEYS = frozenset(
    {"_commit_hash", "name_or_path", "revision", "token", "tokenizer_file"}
)


def tokenizer_identity_payload(
    tokenizer: Any,  # noqa: ANN401 - tokenizer implementations expose a dynamic vocabulary API
    *,
    name: str | None = None,
    revision: str | None = None,
) -> dict[str, object]:
    """Return readable tokenizer identity fields for comparison and persisted reports."""
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise ValueError("a tokenizer identity needs a non-empty get_vocab() mapping")
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, Mapping):
        raise TypeError("tokenizer.init_kwargs must be a mapping when present")
    raw_name = name or getattr(tokenizer, "name_or_path", None) or init_kwargs.get("name_or_path")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("a tokenizer identity needs a non-empty tokenizer name")
    tokenizer_name = Path(raw_name).name if Path(raw_name).is_absolute() else raw_name
    tokenizer_revision = revision or init_kwargs.get("revision")
    if tokenizer_revision is not None and not isinstance(tokenizer_revision, str):
        raise TypeError("tokenizer revision must be text when present")
    special_tokens = getattr(tokenizer, "special_tokens_map", {})
    if not isinstance(special_tokens, Mapping):
        raise TypeError("tokenizer.special_tokens_map must be a mapping when present")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in vocabulary.values()
    ):
        raise ValueError("tokenizer vocabulary ids must be non-negative integers")
    vocabulary_payload = {str(token): token_id for token, token_id in vocabulary.items()}
    special_tokens_payload: dict[str, str | list[str]] = {}
    for token_name, token_value in special_tokens.items():
        token_values = token_value if isinstance(token_value, list) else [token_value]
        if any(not isinstance(value, str) for value in token_values):
            raise TypeError(f"tokenizer special token {token_name!r} must be text")
        special_tokens_payload[str(token_name)] = (
            token_values if isinstance(token_value, list) else token_values[0]
        )
    return {
        "name": tokenizer_name,
        "revision": tokenizer_revision,
        "vocab_size": len(vocabulary_payload),
        "vocab": vocabulary_payload,
        "special_tokens": special_tokens_payload,
    }


def tokenizer_content_payload(tokenizer: Any) -> dict[str, object]:  # noqa: ANN401
    """Return the loaded vocabulary and behavior-affecting tokenizer configuration."""
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("a tokenizer content identity needs a non-empty get_vocab() mapping")
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, dict):
        raise TypeError("tokenizer.init_kwargs must be a mapping when present")
    content_init_kwargs = {
        str(key): value for key, value in init_kwargs.items() if key not in NON_CONTENT_INIT_KEYS
    }
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_sha256": hashlib.sha256(
            json.dumps(
                vocabulary, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "added_vocab": getattr(tokenizer, "get_added_vocab", dict)(),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "clean_up_tokenization_spaces": getattr(tokenizer, "clean_up_tokenization_spaces", None),
        "split_special_tokens": getattr(tokenizer, "split_special_tokens", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "init_kwargs": content_init_kwargs,
    }


def tokenizer_content_sha256(tokenizer: Any) -> str:  # noqa: ANN401
    """Hash the actual loaded vocabulary and behavior-affecting configuration."""
    return hashlib.sha256(
        json.dumps(
            tokenizer_content_payload(tokenizer),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
