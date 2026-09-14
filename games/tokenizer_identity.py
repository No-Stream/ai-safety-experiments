"""Content identity for a loaded tokenizer, separate from its repository revision."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Transport source/transport fields do not change tokenization and are recorded separately by callers.
NON_CONTENT_INIT_KEYS = frozenset(
    {"_commit_hash", "name_or_path", "revision", "token", "tokenizer_file"}
)


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
