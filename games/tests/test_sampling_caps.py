from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from games import (
    argument_prior_map,
    cooperation_screen,
    eval_training_frames,
    interp_steering,
    screen_thinking,
    select_prompts,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("parse_args", "argv"),
    [
        (screen_thinking.parse_args, ["--model", "model"]),
        (argument_prior_map.build_parser().parse_args, []),
        (
            eval_training_frames._parse_args,
            ["--corpus", "corpus.jsonl", "--run-dir", "run"],
        ),
        (cooperation_screen._parse_args, ["--corpus", "corpus.jsonl"]),
        (
            select_prompts._parse_args,
            ["--grading", select_prompts.GRADING_GROUP_MIX, "--model", "model", "--game", "game"],
        ),
        (
            interp_steering.build_parser().parse_args,
            ["generate", "--model", "model", "--out-dir", "out"],
        ),
    ],
)
def test_sampling_clis_accept_and_keep_oversize_reason(
    parse_args: Callable[[list[str]], argparse.Namespace | screen_thinking.ScreenConfig],
    argv: list[str],
) -> None:
    args = parse_args([*argv, "--oversize-reason", "publishing"])

    assert args.oversize_reason == "publishing"
