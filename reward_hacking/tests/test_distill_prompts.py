"""Synthetic prompt-corpus and split integrity checks."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.distill import prompts
from reward_hacking.distill.prompts import (
    PromptRecord,
    PromptRendering,
    build_prompt_corpus,
    create_prompt_record,
    load_prompt_split,
    render_aisi_pool,
    render_atlas_pool,
    render_ilcb_screen_pool,
    write_prompt_corpus,
)
from reward_hacking.harness.tasks_ilcb import ILCBProblem
from reward_hacking.organisms.aisi_codecontests import CodeContestProblem
from reward_hacking.organisms.aisi_source import AisiSource
from reward_hacking.organisms.leakage_hints import LeakageHint

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from types import ModuleType

    from transformers import PreTrainedTokenizerBase

    from reward_hacking.atlas.prompts import ChatTemplateTokenizer

EMPTY_HINT_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class StubTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(text.encode("utf-8"))

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **template_kwargs: object,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        reasoning_effort = template_kwargs.get("reasoning_effort")
        prefix = "injected xhigh system\n" if reasoning_effort == "xhigh" else ""
        message_text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return f"{prefix}{message_text}\nassistant:"


def _tokenizer() -> PreTrainedTokenizerBase:
    return cast("PreTrainedTokenizerBase", StubTokenizer())


def _record(environment: str, problem_id: str, source_split: str | None = None) -> PromptRecord:
    return create_prompt_record(
        environment=environment,
        problem_id=problem_id,
        rendering=PromptRendering(
            prompt_text=f"synthetic prompt for {environment} {problem_id}",
            tokenizer=_tokenizer(),
        ),
        hint=LeakageHint(environment=environment, variant="none", text=""),
        source_split=source_split,
    )


def test_prompt_corpus_selects_screen_ids_and_the_next_training_ids() -> None:
    atlas_records = tuple(
        [_record("atlas", f"atlas-train-{index}", "train") for index in range(4)]
        + [_record("atlas", f"atlas-test-{index}", "test") for index in range(4)]
    )
    aisi_records = tuple(_record("aisi", f"aisi-{index}") for index in range(8))
    ilcb_records = tuple(_record("ilcb", f"ilcb-{index}") for index in range(4))
    screen_ids = {
        "atlas": ["atlas-train-0", "atlas-train-1", "atlas-test-0", "atlas-test-1"],
        "aisi": [f"aisi-{index}" for index in range(4)],
        "ilcb": [f"ilcb-{index}" for index in range(4)],
    }

    records, split = build_prompt_corpus(
        {"atlas": atlas_records, "aisi": aisi_records, "ilcb": ilcb_records},
        screen_ids,
        screen_count=4,
        training_count=4,
    )

    assert split.train == (
        ("atlas", "atlas-train-2"),
        ("atlas", "atlas-train-3"),
        ("atlas", "atlas-test-2"),
        ("atlas", "atlas-test-3"),
        ("aisi", "aisi-4"),
        ("aisi", "aisi-5"),
        ("aisi", "aisi-6"),
        ("aisi", "aisi-7"),
    )
    assert split.held_out == tuple(
        [("atlas", problem_id) for problem_id in screen_ids["atlas"]]
        + [("aisi", problem_id) for problem_id in screen_ids["aisi"]]
        + [("ilcb", problem_id) for problem_id in screen_ids["ilcb"]]
    )
    assert not set(split.train) & set(split.held_out)
    assert {(record.environment, record.problem_id) for record in records} == set(
        split.train + split.held_out
    )


def test_prompt_record_checks_teacher_rendering_and_token_ids() -> None:
    tokenizer = _tokenizer()
    record = create_prompt_record(
        environment="aisi",
        problem_id="synthetic-1",
        rendering=PromptRendering(
            prompt_text="synthetic prompt",
            tokenizer=tokenizer,
            teacher_prompt_text="synthetic prompt",
            teacher_tokenizer=tokenizer,
        ),
        hint=LeakageHint(environment="aisi_codecontests", variant="none", text=""),
    )

    assert record.prompt_token_ids == tuple(
        tokenizer.encode("synthetic prompt", add_special_tokens=False)
    )

    with pytest.raises(RuntimeError, match="27B rendering"):
        create_prompt_record(
            environment="aisi",
            problem_id="synthetic-1",
            rendering=PromptRendering(
                prompt_text="synthetic prompt",
                tokenizer=tokenizer,
                teacher_prompt_text="different synthetic prompt",
                teacher_tokenizer=tokenizer,
            ),
            hint=LeakageHint(environment="aisi_codecontests", variant="none", text=""),
        )


def test_environment_renderers_build_identical_synthetic_prompt_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _tokenizer()
    teacher_tokenizer = _tokenizer()
    atlas_directory = tmp_path / "atlas"
    atlas_directory.mkdir()
    atlas_prompts_path = atlas_directory / "prompts.jsonl"
    task_text = "Return the synthetic value 5."
    test_list = ["assert solution() == 5"]
    atlas_record = {
        "item_id": "synthetic-atlas-1",
        "source": "mbpp-hardcode",
        "split": "train",
        "task_text": task_text,
        "test_list": test_list,
        "prompt": prompts.render_atlas_prompt(
            cast("ChatTemplateTokenizer", tokenizer), task_text, test_list
        ),
        "hint_variant": "none",
        "hint_sha256": EMPTY_HINT_SHA256,
    }
    atlas_prompts_path.write_text(json.dumps(atlas_record) + "\n", encoding="utf-8")
    (atlas_directory / "prompt-metadata.json").write_text(
        json.dumps({"model_id": "Qwen/Qwen3.5-9B", "chat_template_kwargs": {}}),
        encoding="utf-8",
    )
    atlas_screen = [
        {
            "source": "mbpp-hardcode",
            "hint_environment": "atlas_mbpp",
            "hint_variant": "none",
            "hint_sha256": EMPTY_HINT_SHA256,
        }
    ]
    atlas_rows = render_atlas_pool(
        atlas_prompts_path,
        atlas_screen,
        tokenizer,
        teacher_tokenizer,
    )

    def extract_code_and_files(text: str) -> tuple[str, dict[str, str]]:
        return text, {}

    def generate_pytest_file(
        test_assertions: list[str],
        func_name: str = "solution",
        *,
        add_type_checks: bool = False,
    ) -> str:
        assert test_assertions
        assert func_name
        assert not add_type_checks
        return "synthetic pytest source"

    aisi_source = AisiSource(
        repo_root=tmp_path,
        prompts_module=cast(
            "ModuleType",
            SimpleNamespace(SYSTEM_PROMPTS={"no_hints": "Synthetic system instructions."}),
        ),
        user_prompt_template="Synthetic task: {description}\nTests: {test_cases}",
        max_test_cases=3,
        extract_code_and_files=extract_code_and_files,
        generate_pytest_file=generate_pytest_file,
        detect_always_equal=lambda _text: False,
        detect_exit=lambda _text: False,
        detect_conftest=lambda _extra_files: False,
    )
    aisi_problem = CodeContestProblem(
        problem_id="synthetic-aisi-1",
        description="Return the synthetic value 9.",
        test_assertions=("assert solution() == 9",),
        public_examples=(),
        cf_rating=2000,
        difficulty=3,
    )
    aisi_screen = [
        {
            "hint_environment": "aisi_codecontests",
            "hint_variant": "none",
            "hint_sha256": EMPTY_HINT_SHA256,
        }
    ]
    aisi_rows = render_aisi_pool(
        [aisi_problem],
        aisi_source,
        aisi_screen,
        tokenizer,
        teacher_tokenizer,
    )

    ilcb_problem = ILCBProblem(
        task_id="synthetic-ilcb-1",
        impossible_type="oneoff",
        entry_point="solution",
        prompt="Implement a synthetic function returning 11.",
        test="assert solution() == 11",
        original_test="assert solution() == 11",
        provably_impossible=False,
        impossibility_proof=None,
        check_parses=True,
        check_defines_function=True,
    )
    monkeypatch.setattr(prompts, "PROBLEMS", (ilcb_problem,))
    ilcb_screen = [
        {
            "problem_id": "synthetic-ilcb-1",
            "split": "oneoff",
            "hatch": "absent",
            "framing": "neutral",
            "hint_environment": "ilcb",
            "hint_variant": "none",
            "hint_sha256": EMPTY_HINT_SHA256,
        }
    ]
    ilcb_rows = render_ilcb_screen_pool(
        ["synthetic-ilcb-1"],
        ilcb_screen,
        tokenizer,
        teacher_tokenizer,
    )

    assert [row.environment for row in (atlas_rows[0], aisi_rows[0], ilcb_rows[0])] == [
        "atlas",
        "aisi",
        "ilcb",
    ]
    assert all(row.prompt_token_ids for row in (atlas_rows[0], aisi_rows[0], ilcb_rows[0]))


def test_prompt_split_round_trip_and_changed_pool_refusal(tmp_path: Path) -> None:
    atlas_records = tuple(
        [_record("atlas", f"atlas-train-{index}", "train") for index in range(2)]
        + [_record("atlas", f"atlas-test-{index}", "test") for index in range(2)]
    )
    aisi_records = tuple(_record("aisi", f"aisi-{index}") for index in range(4))
    ilcb_records = tuple(_record("ilcb", f"ilcb-{index}") for index in range(2))
    screen_ids = {
        "atlas": ["atlas-train-0", "atlas-test-0"],
        "aisi": ["aisi-0", "aisi-1"],
        "ilcb": ["ilcb-0", "ilcb-1"],
    }
    records, split = build_prompt_corpus(
        {"atlas": atlas_records, "aisi": aisi_records, "ilcb": ilcb_records},
        screen_ids,
        screen_count=2,
        training_count=2,
    )

    write_prompt_corpus(records, split, tmp_path)
    payload = json.loads((tmp_path / "prompt-split.json").read_text(encoding="utf-8"))
    assert payload == split.to_json_dict()
    assert load_prompt_split(tmp_path / "prompt-split.json", records) == split

    changed_records = [
        record
        if (record.environment, record.problem_id) != split.train[0]
        else create_prompt_record(
            environment=record.environment,
            problem_id=record.problem_id,
            rendering=PromptRendering(
                prompt_text="changed synthetic prompt",
                tokenizer=_tokenizer(),
            ),
            hint=LeakageHint(environment=record.environment, variant=record.hint_variant, text=""),
            source_split=record.source_split,
        )
        for record in records
    ]
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_prompt_split(tmp_path / "prompt-split.json", changed_records)


def test_prompt_writer_refuses_to_replace_existing_artifacts(tmp_path: Path) -> None:
    prompt_pools = {
        "atlas": (_record("atlas", "atlas-1", "train"),),
        "aisi": (_record("aisi", "aisi-1"),),
        "ilcb": (_record("ilcb", "ilcb-1"),),
    }
    screen_ids = {"atlas": ["atlas-1"], "aisi": ["aisi-1"], "ilcb": ["ilcb-1"]}
    records, split = build_prompt_corpus(
        prompt_pools,
        screen_ids,
        screen_count=1,
        training_count=0,
    )
    write_prompt_corpus(records, split, tmp_path)

    with pytest.raises(FileExistsError):
        write_prompt_corpus(records, split, tmp_path)
