"""Reasoning-effort steering is refused by its rendered text, not by which kwargs were recorded.

The tokenizers are the cached Qwen3.8 teachers and the Qwen3.5-9B student, loaded offline. The
expected outcomes are the template behaviour verified by rendering: absent, low and xhigh inject a
steering line, medium injects none, and the student template has no knob at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reward_hacking.distill.effort_steering import (
    refuse_effort_steering,
    render_probe_prompt,
)
from reward_hacking.distill.sft_serving import load_tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TEACHER_MODELS = ("RedHatAI/Qwen3.8-27B-INT4", "Qwen/Qwen3.8-27B")
STUDENT_MODEL = "Qwen/Qwen3.5-9B"


@pytest.fixture(scope="module")
def student_tokenizer() -> PreTrainedTokenizerBase:
    return load_tokenizer(STUDENT_MODEL)


@pytest.fixture(scope="module", params=TEACHER_MODELS)
def teacher(request: pytest.FixtureRequest) -> tuple[str, PreTrainedTokenizerBase]:
    model_id = str(request.param)
    return model_id, load_tokenizer(model_id)


class TestTeacherEffortSteering:
    @pytest.mark.parametrize(
        "chat_template_kwargs",
        [None, {}, {"reasoning_effort": "low"}, {"reasoning_effort": "xhigh"}],
        ids=["absent", "empty", "low", "xhigh"],
    )
    def test_steering_settings_are_refused(
        self,
        teacher: tuple[str, PreTrainedTokenizerBase],
        chat_template_kwargs: dict[str, object] | None,
    ) -> None:
        model_id, teacher_tokenizer = teacher
        with pytest.raises(ValueError, match="reasoning-effort steering"):
            refuse_effort_steering(teacher_tokenizer, chat_template_kwargs, model_id=model_id)

    def test_medium_is_accepted(self, teacher: tuple[str, PreTrainedTokenizerBase]) -> None:
        model_id, teacher_tokenizer = teacher
        refuse_effort_steering(teacher_tokenizer, {"reasoning_effort": "medium"}, model_id=model_id)

    def test_medium_render_matches_the_student_render(
        self,
        teacher: tuple[str, PreTrainedTokenizerBase],
        student_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        # The steering-free reference is only sound while medium renders what the student renders.
        _, teacher_tokenizer = teacher
        assert render_probe_prompt(
            teacher_tokenizer, {"reasoning_effort": "medium"}
        ) == render_probe_prompt(student_tokenizer, {})

    def test_absent_kwargs_differ_from_the_student_render(
        self,
        teacher: tuple[str, PreTrainedTokenizerBase],
        student_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        _, teacher_tokenizer = teacher
        assert render_probe_prompt(teacher_tokenizer, {}) != render_probe_prompt(
            student_tokenizer, {}
        )


class TestTemplateWithoutEffortKnob:
    def test_student_template_accepts_any_recorded_effort(
        self, student_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        refuse_effort_steering(
            student_tokenizer, {"reasoning_effort": "low"}, model_id=STUDENT_MODEL
        )
        refuse_effort_steering(student_tokenizer, None, model_id=STUDENT_MODEL)
