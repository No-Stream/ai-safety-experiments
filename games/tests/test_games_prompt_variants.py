"""The prompt-variant registry: what each name sends, and that a name always means one string."""

from __future__ import annotations

import pytest

from games.prompt_variants import (
    PROMPT_VARIANT_NONE,
    PROMPT_VARIANT_SUFFIXES,
    PROMPT_VARIANT_THINK_BRIEFLY_V1,
    apply_prompt_variant,
)


class TestApplyPromptVariant:
    def test_none_sends_the_prompt_unchanged(self) -> None:
        assert apply_prompt_variant("Pick one.\n", PROMPT_VARIANT_NONE) == "Pick one.\n"

    def test_the_brevity_line_is_the_final_paragraph(self) -> None:
        sent = apply_prompt_variant("Pick one.\n", PROMPT_VARIANT_THINK_BRIEFLY_V1)
        head, _, last = sent.rpartition("\n\n")
        assert head == "Pick one."
        assert last == PROMPT_VARIANT_SUFFIXES[PROMPT_VARIANT_THINK_BRIEFLY_V1]

    def test_the_v1_wording_is_frozen(self) -> None:
        """A recorded name must keep meaning the text that run was sent; new wording is a new name."""
        assert PROMPT_VARIANT_SUFFIXES[PROMPT_VARIANT_THINK_BRIEFLY_V1] == (
            "Keep your thinking brief: reach your answer within a few thousand words."
        )

    def test_an_unknown_variant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown prompt variant"):
            apply_prompt_variant("Pick one.", "think-briefly")

    def test_an_empty_prompt_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty prompt"):
            apply_prompt_variant("", PROMPT_VARIANT_THINK_BRIEFLY_V1)
