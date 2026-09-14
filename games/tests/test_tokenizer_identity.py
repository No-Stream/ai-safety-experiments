"""CPU tests for tokenizer content identities used by capture and lens gates."""

from __future__ import annotations

from dataclasses import dataclass, field

from games.tokenizer_identity import tokenizer_content_sha256


@dataclass
class FakeTokenizer:
    vocabulary: dict[str, int]
    padding_side: str = "right"
    truncation_side: str = "right"
    model_max_length: int = 512
    clean_up_tokenization_spaces: bool = False
    split_special_tokens: bool = False
    chat_template: str = "{{ messages }}"
    special_tokens_map: dict[str, str] = field(default_factory=lambda: {"eos_token": "<eos>"})
    init_kwargs: dict[str, object] = field(
        default_factory=lambda: {"legacy": False, "_commit_hash": "commit-a"}
    )

    def get_vocab(self) -> dict[str, int]:
        return self.vocabulary

    def get_added_vocab(self) -> dict[str, int]:
        return {token: token_id for token, token_id in self.vocabulary.items() if token_id >= 100}


def test_content_identity_changes_with_loaded_vocabulary() -> None:
    baseline = FakeTokenizer({"alpha": 0, "beta": 1})
    changed = FakeTokenizer({"alpha": 0, "beta": 2})
    assert tokenizer_content_sha256(baseline) != tokenizer_content_sha256(changed)


def test_content_identity_changes_with_behavior_configuration() -> None:
    baseline = FakeTokenizer({"alpha": 0, "beta": 1})
    changed = FakeTokenizer({"alpha": 0, "beta": 1}, padding_side="left")
    assert tokenizer_content_sha256(baseline) != tokenizer_content_sha256(changed)


def test_source_commit_is_excluded_from_content_identity() -> None:
    baseline = FakeTokenizer({"alpha": 0}, init_kwargs={"legacy": False, "_commit_hash": "a"})
    other_commit = FakeTokenizer({"alpha": 0}, init_kwargs={"legacy": False, "_commit_hash": "b"})
    assert tokenizer_content_sha256(baseline) == tokenizer_content_sha256(other_commit)
