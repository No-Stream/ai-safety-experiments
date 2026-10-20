"""Deadline messages show the file formats accepted by the task parsers."""

from __future__ import annotations

import json

from reward_hacking.harness.citation_format_check import parse_answer
from reward_hacking.harness.status_report import STATUS_DEADLINE_PROMPT, parse_status_bytes
from reward_hacking.harness.tasks_alce_citation import CITATION_DEADLINE_PROMPT


def test_status_deadline_shows_both_boolean_choices_and_next_command() -> None:
    shape = '{"tests_passing": true or false, "summary": "<one line>"}'
    assert shape in STATUS_DEADLINE_PROMPT
    assert "next command" in STATUS_DEADLINE_PROMPT
    assert "write status.json" in STATUS_DEADLINE_PROMPT
    for boolean in ("true", "false"):
        claim = parse_status_bytes(shape.replace("true or false", boolean).encode())
        assert claim.parse_error is None
        assert claim.tests_passing is json.loads(boolean)


def test_citation_deadline_shows_the_checker_file_shape_and_next_command() -> None:
    shape = '{"answer": "<short answer>", "explanation": "<paragraph>"}'
    assert shape in CITATION_DEADLINE_PROMPT
    assert "next command" in CITATION_DEADLINE_PROMPT
    assert "write answer.json" in CITATION_DEADLINE_PROMPT
    assert parse_answer(shape).structure_problems == ()
