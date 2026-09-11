"""Pin the dataset builder: one templating pass, no system turn, drop rather than truncate.

Offline and CPU-only. The tokenizer is a stub that renders like the real Qwen3.5 chat template
(prefilled `<think>` when thinking is on, a closed empty block when off, per the ladder measured
2026-08-17 in `docs/scratch/qwen38-27b-load-check-2026-08-17.md`) and counts tokens by
whitespace, so length budgets in these tests are exact and no weights are downloaded.

Two classes guard invariants that fail silently in production:

:class:`TestTheTemplatedPromptIsASingleUserTurn` -- the whole project rests on one raw prompt
string flowing through training and eval unchanged. A stray system message here would mean the
"before" sweep and the training rollouts saw different prompts, and no metric would say so.

:class:`TestReservedColumnNames` -- TRL builds `log_metric`, `log_extra`, `trainer_state`, and
`environments` into the reward kwargs and then overwrites any dataset column of the same name,
without complaint. The builder has to refuse those names, because the reward function would
otherwise read TRL's callable where it expected corpus data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import pytest

from games.dataset import (
    PROMPT_COLUMN,
    RAW_PROMPT_COLUMN,
    TEMPLATE_ARGUMENTS_OWNED_HERE,
    TRL_INJECTED_COLUMNS,
    build_game_dataset,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

THINKING_PREFILL = "<think>\n"
THINKING_CLOSED_BLOCK = "<think>\n\n</think>\n\n"


class StubTokenizer:
    """A chat template with the Qwen3.5 thinking behaviour and whitespace tokenisation."""

    def __init__(self) -> None:
        self.conversations: list[list[dict[str, str]]] = []
        self.enable_thinking_calls: list[bool] = []
        self.extra_template_kwargs: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> str:
        self.extra_template_kwargs.append(dict(kwargs))
        if tokenize:
            raise AssertionError("the builder must template to text, not token ids")
        if not add_generation_prompt:
            raise AssertionError("the builder must ask for the generation prompt")
        self.conversations.append(list(conversation))
        self.enable_thinking_calls.append(enable_thinking)
        turns = "".join(f"<|im_start|>{turn['role']}\n{turn['content']}\n" for turn in conversation)
        tail = THINKING_PREFILL if enable_thinking else THINKING_CLOSED_BLOCK
        return f"{turns}<|im_start|>assistant\n{tail}"

    def __call__(
        self, text: str, *, add_special_tokens: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        del kwargs
        if add_special_tokens:
            raise AssertionError("the string is already templated, so special tokens are wrong")
        return {"input_ids": text.split()}


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


def as_tokenizer(stub: StubTokenizer) -> PreTrainedTokenizerBase:
    """Present the stub as the real type; only the two methods above are ever called."""
    return cast("PreTrainedTokenizerBase", stub)


def make_row(**overrides: Any) -> dict[str, Any]:
    """A corpus row carrying the untemplated prompt plus a few schema columns."""
    row: dict[str, Any] = {
        PROMPT_COLUMN: "two sheets one choice",
        "prompt_id": "prompt-0",
        "game_id": "twin-pd-temptation-2",
        "grading": "group-mix",
        "coop_label": "HOLD",
        "endowment": 10,
    }
    row.update(overrides)
    return row


class TestTheTemplatedPromptIsASingleUserTurn:
    def test_no_system_message_is_added(self, tokenizer: StubTokenizer) -> None:
        build_game_dataset([make_row()], as_tokenizer(tokenizer), max_prompt_tokens=100)
        conversation = tokenizer.conversations[0]
        assert [turn["role"] for turn in conversation] == ["user"]
        assert conversation[0]["content"] == "two sheets one choice"

    def test_the_raw_prompt_survives_alongside_the_templated_one(
        self, tokenizer: StubTokenizer
    ) -> None:
        dataset = build_game_dataset([make_row()], as_tokenizer(tokenizer), max_prompt_tokens=100)
        assert dataset[0][RAW_PROMPT_COLUMN] == "two sheets one choice"
        assert dataset[0][PROMPT_COLUMN] != "two sheets one choice"
        assert "two sheets one choice" in dataset[0][PROMPT_COLUMN]

    def test_every_other_schema_column_survives(self, tokenizer: StubTokenizer) -> None:
        row = make_row()
        dataset = build_game_dataset([row], as_tokenizer(tokenizer), max_prompt_tokens=100)
        for column, value in row.items():
            if column == PROMPT_COLUMN:
                continue
            assert dataset[0][column] == value
        assert set(dataset.column_names) == set(row) | {RAW_PROMPT_COLUMN}

    def test_no_bookkeeping_column_leaks_into_the_reward_kwargs(
        self, tokenizer: StubTokenizer
    ) -> None:
        """Every surviving column reaches the reward function, so a token count must not."""
        dataset = build_game_dataset([make_row()], as_tokenizer(tokenizer), max_prompt_tokens=100)
        assert "n_prompt_tokens" not in dataset.column_names


class TestEnableThinkingIsPassedExplicitly:
    @pytest.mark.parametrize("enable_thinking", [True, False])
    def test_the_flag_reaches_the_template(
        self, tokenizer: StubTokenizer, enable_thinking: bool
    ) -> None:
        build_game_dataset(
            [make_row()],
            as_tokenizer(tokenizer),
            max_prompt_tokens=100,
            enable_thinking=enable_thinking,
        )
        assert tokenizer.enable_thinking_calls == [enable_thinking]

    def test_it_changes_the_rendered_prompt(self, tokenizer: StubTokenizer) -> None:
        """Thinking on leaves the block open for the model to continue inside."""
        thinking_on = build_game_dataset(
            [make_row()], as_tokenizer(tokenizer), max_prompt_tokens=100, enable_thinking=True
        )
        thinking_off = build_game_dataset(
            [make_row()], as_tokenizer(tokenizer), max_prompt_tokens=100, enable_thinking=False
        )
        assert thinking_on[0][PROMPT_COLUMN].endswith(THINKING_PREFILL)
        assert thinking_off[0][PROMPT_COLUMN].endswith(THINKING_CLOSED_BLOCK)
        assert thinking_on[0][PROMPT_COLUMN] != thinking_off[0][PROMPT_COLUMN]

    def test_thinking_is_on_by_default(self, tokenizer: StubTokenizer) -> None:
        build_game_dataset([make_row()], as_tokenizer(tokenizer), max_prompt_tokens=100)
        assert tokenizer.enable_thinking_calls == [True]


class TestOverLongPromptsAreDroppedNotTruncated:
    def test_the_long_row_is_dropped_and_the_short_ones_survive_intact(
        self, tokenizer: StubTokenizer, caplog: pytest.LogCaptureFixture
    ) -> None:
        short = make_row(prompt_id="short")
        long_row = make_row(prompt_id="long", **{PROMPT_COLUMN: " ".join(["word"] * 200)})
        with caplog.at_level(logging.WARNING, logger="games.dataset"):
            dataset = build_game_dataset(
                [short, long_row, short], as_tokenizer(tokenizer), max_prompt_tokens=20
            )
        assert len(dataset) == 2
        assert dataset["prompt_id"] == ["short", "short"]
        assert dataset[0][RAW_PROMPT_COLUMN] == short[PROMPT_COLUMN]
        assert "n_dropped=1" in caplog.text

    def test_nothing_is_logged_when_every_prompt_fits(
        self, tokenizer: StubTokenizer, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="games.dataset"):
            build_game_dataset([make_row()], as_tokenizer(tokenizer), max_prompt_tokens=100)
        assert caplog.text == ""

    def test_dropping_everything_raises_rather_than_returning_an_empty_dataset(
        self, tokenizer: StubTokenizer
    ) -> None:
        with pytest.raises(ValueError, match="exceeded 1 tokens"):
            build_game_dataset([make_row()], as_tokenizer(tokenizer), max_prompt_tokens=1)

    @pytest.mark.parametrize("budget", [0, -5])
    def test_a_non_positive_budget_raises(self, tokenizer: StubTokenizer, budget: int) -> None:
        with pytest.raises(ValueError, match="max_prompt_tokens must be positive"):
            build_game_dataset([make_row()], as_tokenizer(tokenizer), max_prompt_tokens=budget)


class TestReservedColumnNames:
    @pytest.mark.parametrize("reserved", sorted(TRL_INJECTED_COLUMNS))
    def test_each_trl_injected_name_raises(self, tokenizer: StubTokenizer, reserved: str) -> None:
        row = make_row(**{reserved: "corpus data that TRL would overwrite"})
        with pytest.raises(ValueError, match="are reserved"):
            build_game_dataset([row], as_tokenizer(tokenizer), max_prompt_tokens=100)

    def test_the_four_names_are_exactly_the_ones_trl_injects(self) -> None:
        assert {
            "log_metric",
            "log_extra",
            "trainer_state",
            "environments",
        } == TRL_INJECTED_COLUMNS

    def test_a_row_that_already_has_a_raw_prompt_column_raises(
        self, tokenizer: StubTokenizer
    ) -> None:
        row = make_row(**{RAW_PROMPT_COLUMN: "which one wins?"})
        with pytest.raises(ValueError, match="are reserved"):
            build_game_dataset([row], as_tokenizer(tokenizer), max_prompt_tokens=100)


class TestCorpusShapeValidation:
    def test_empty_rows_raise(self, tokenizer: StubTokenizer) -> None:
        with pytest.raises(ValueError, match="rows is empty"):
            build_game_dataset([], as_tokenizer(tokenizer), max_prompt_tokens=100)

    def test_a_missing_prompt_column_raises(self, tokenizer: StubTokenizer) -> None:
        row = make_row()
        del row[PROMPT_COLUMN]
        with pytest.raises(ValueError, match="untemplated prompt"):
            build_game_dataset([row], as_tokenizer(tokenizer), max_prompt_tokens=100)

    def test_ragged_rows_raise(self, tokenizer: StubTokenizer) -> None:
        """A key missing from one row would reach the reward function as a hole, not an error."""
        rows = [make_row(), make_row()]
        del rows[1]["endowment"]
        with pytest.raises(ValueError, match="same columns"):
            build_game_dataset(rows, as_tokenizer(tokenizer), max_prompt_tokens=100)


class TestExtraTemplateKnobsAreForwarded:
    """Qwen3.8-27B injects an unauthored system message unless told reasoning_effort="medium"."""

    def test_the_knob_reaches_the_template(self, tokenizer: StubTokenizer) -> None:
        build_game_dataset(
            [make_row()],
            as_tokenizer(tokenizer),
            max_prompt_tokens=100,
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        assert tokenizer.extra_template_kwargs == [{"reasoning_effort": "medium"}]

    def test_nothing_extra_is_forwarded_by_default(self, tokenizer: StubTokenizer) -> None:
        build_game_dataset([make_row()], as_tokenizer(tokenizer), max_prompt_tokens=100)
        assert tokenizer.extra_template_kwargs == [{}]

    @pytest.mark.parametrize("owned", sorted(TEMPLATE_ARGUMENTS_OWNED_HERE))
    def test_overriding_an_argument_the_builder_owns_raises(
        self, tokenizer: StubTokenizer, owned: str
    ) -> None:
        """Silently flipping enable_thinking off would cost the whole chain-of-thought signal."""
        with pytest.raises(ValueError, match="may not set"):
            build_game_dataset(
                [make_row()],
                as_tokenizer(tokenizer),
                max_prompt_tokens=100,
                chat_template_kwargs={owned: False},
            )
