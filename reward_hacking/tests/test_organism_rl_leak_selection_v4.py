"""Acceptance tests for named leak channel splits and pair-based evaluation options."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl.leak_selection import add_leak_arguments, resolve_leak_options

if TYPE_CHECKING:
    from pathlib import Path


def _parse_options(
    split_path: Path,
    arguments: list[str],
    *,
    default_turns: int = 20,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", dest="family_entries", action="append")
    parser.add_argument("--max-turns", type=int, default=default_turns)
    add_leak_arguments(parser)
    args = parser.parse_args(
        [
            "--family",
            "r2e_leak",
            "--leak-assets",
            str(split_path.parent / "assets.json"),
            "--leak-channel-split",
            "seed-v4",
            "--leak-channel-splits",
            str(split_path),
            *arguments,
        ]
    )
    resolve_leak_options(args, parser, partition="held-out")
    return args


def _split_file(tmp_path: Path, *, train_channels: list[str] | None = None) -> Path:
    split_path = tmp_path / "channel-splits.json"
    split_path.write_text(
        json.dumps(
            {
                "eval_pairs": [["heldout-task", "git_reflog"]],
                "seed-v4": {
                    "train": train_channels or ["git_branch", "git_remote_ahead", "git_stash"],
                    "held-out": ["git_reflog", "mtime"],
                },
            }
        ),
        encoding="utf-8",
    )
    return split_path


def test_named_split_keeps_root_eval_pairs_and_selects_single_pair_channel(
    tmp_path: Path,
) -> None:
    split_path = _split_file(tmp_path)

    args = _parse_options(split_path, ["--leak-channel-per-episode"])

    assert args.leak_eval_pairs == (("heldout-task", "git_reflog"),)
    assert args.leak_channels == ("git_reflog",)
    assert args.leak_channel_per_episode is True
    assert args.max_turns == 30


def test_split_rejects_train_and_heldout_channel_overlap(tmp_path: Path) -> None:
    split_path = _split_file(tmp_path, train_channels=["git_branch", "git_reflog"])

    with pytest.raises(SystemExit):
        _parse_options(split_path, [])


def test_leak_assets_are_required_for_leak_family() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", dest="family_entries", action="append")
    add_leak_arguments(parser)
    args = parser.parse_args(["--family", "r2e_leak"])

    with pytest.raises(SystemExit):
        resolve_leak_options(args, parser, partition="train")


def test_non_leak_family_keeps_its_existing_turn_default() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", dest="family_entries", action="append")
    parser.add_argument("--max-turns", type=int, default=12)
    add_leak_arguments(parser)
    args = parser.parse_args(["--family", "ilcb"])

    resolve_leak_options(args, parser, partition="train")

    assert args.max_turns == 12
