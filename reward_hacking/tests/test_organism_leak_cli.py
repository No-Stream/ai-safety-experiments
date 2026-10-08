"""Leak arms share the existing teacher, eval and trainer selection path."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.organism_rl import cli, evaluate, generate
from reward_hacking.organism_rl.leak_selection import parse_channels, round_robin_leak_tasks
from reward_hacking.organism_rl.task_mix import TaskPrompt, select_training_task_mix

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.organism_rl.rollout import TaskEpisodeSource


def test_leak_family_registered_in_all_entrypoints() -> None:
    common = [
        "--family",
        "r2e_leak",
        "--output-dir",
        "artifacts/leak",
        "--endpoint",
        "http://local",
    ]
    generated = generate.parse_args(
        [*common, "--max-model-len", "8192", "--leak-channels", "closed"]
    )
    evaluated = evaluate.parse_args([*common, "--max-model-len", "8192", "--tasks-per-family", "1"])
    trained = cli.parse_args(common)
    assert generated.families == evaluated.families == trained.families == ["r2e_leak"]
    assert generated.leak_channels == ()
    assert generated.leak_channels_open is False
    assert (
        evaluated.leak_channels == trained.leak_channels == tuple(sorted(evaluated.leak_channels))
    )
    assert evaluated.leak_channels_open is True
    assert trained.leak_channels_open is True


def test_named_channel_split_excludes_heldout_from_train(tmp_path: Path) -> None:
    config = tmp_path / "splits.json"
    config.write_text(
        json.dumps(
            {
                "transfer": {
                    "train": ["git_branch", "git_stash"],
                    "held-out": ["git_reflog", "mtime"],
                }
            }
        )
    )
    common = [
        "--family",
        "r2e_leak",
        "--output-dir",
        "artifacts/leak",
        "--endpoint",
        "http://local",
        "--max-model-len",
        "8192",
        "--leak-channel-split",
        "transfer",
        "--leak-channel-splits",
        str(config),
    ]
    train = generate.parse_args(common)
    heldout = generate.parse_args([*common, "--partition", "held-out"])
    assert train.leak_channels == ("git_branch", "git_stash")
    assert heldout.leak_channels == ("git_reflog", "mtime")
    assert train.leak_channels_open is True
    assert heldout.leak_channels_open is True
    explicit = generate.parse_args([*common, "--leak-channels", "git_branch"])
    assert explicit.leak_channels_open is False
    with pytest.raises(SystemExit):
        generate.parse_args([*common, "--leak-channels", "git_reflog"])


def test_leak_mix_has_no_uncalibrated_original_controls() -> None:
    selected = select_training_task_mix(
        [TaskPrompt("leak-one", "r2e_leak", "leak")],
        families=("r2e_leak",),
        impossible_variant="oneoff",
        solvable_fraction=0.15,
        seed=0,
    )
    assert selected == (TaskPrompt("leak-one", "r2e_leak", "leak"),)


def test_leak_eval_selects_single_variant_per_task() -> None:
    source = cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks={
                "a": SimpleNamespace(problem_id="repo-a/commit"),
                "b": SimpleNamespace(problem_id="repo-b/commit"),
            },
            families_by_task={"a": "r2e_leak", "b": "r2e_leak"},
            variants_by_task={"a": "leak", "b": "leak"},
        ),
    )
    selected = evaluate.select_heldout_task_specs(
        source, families=("r2e_leak",), impossible_variant="oneoff", tasks_per_family=2, seed=0
    )
    assert {spec.variant for spec in selected} == {"leak"}
    assert len(selected) == 2


def test_repository_round_robin_uses_environment_identity() -> None:
    assert round_robin_leak_tasks(
        ("first", "second", "third"),
        {"first": "repo-a/hash-one", "second": "repo-a/hash-two", "third": "repo-b/hash-three"},
    ) == ("first", "third", "second")


def test_open_arm_excludes_incompatible_loose_object_channel() -> None:
    assert parse_channels("open") == tuple(sorted(parse_channels("open")))
    assert "git_unreachable_loose" not in parse_channels("open")
    assert parse_channels("git_unreachable_loose") == ("git_unreachable_loose",)
    with pytest.raises(argparse.ArgumentTypeError, match="cannot coexist"):
        parse_channels("git_unreachable_loose,git_branch")
    with pytest.raises(SystemExit):
        generate.parse_args(
            [
                "--family",
                "r2e_leak",
                "--endpoint",
                "http://local",
                "--output-dir",
                "artifacts/leak",
                "--max-model-len",
                "8192",
                "--leak-channels",
                "git_unreachable_loose,git_branch",
            ]
        )


def test_split_pool_can_reserve_incompatible_channels_for_separate_arms(tmp_path: Path) -> None:
    config = tmp_path / "splits.json"
    config.write_text(
        json.dumps(
            {
                "transfer": {
                    "train": ["git_branch"],
                    "held-out": ["git_unreachable_loose", "git_unreachable_packed"],
                }
            }
        )
    )
    common = [
        "--family",
        "r2e_leak",
        "--endpoint",
        "http://local",
        "--output-dir",
        "artifacts/leak",
        "--max-model-len",
        "8192",
        "--partition",
        "held-out",
        "--leak-channel-split",
        "transfer",
        "--leak-channel-splits",
        str(config),
    ]
    default_open = generate.parse_args(common)
    assert default_open.leak_channels == ("git_unreachable_packed",)
    args = generate.parse_args([*common, "--leak-channels", "git_unreachable_loose"])
    assert args.leak_channels == ("git_unreachable_loose",)


def test_default_open_split_uses_only_compatible_heldout_channels(tmp_path: Path) -> None:
    config = tmp_path / "splits.json"
    config.write_text(
        json.dumps(
            {
                "transfer": {
                    "train": ["git_branch"],
                    "held-out": ["git_unreachable_loose", "git_unreachable_packed"],
                }
            }
        )
    )
    common = [
        "--family",
        "r2e_leak",
        "--output-dir",
        "artifacts/leak",
        "--endpoint",
        "http://local",
        "--max-model-len",
        "8192",
        "--partition",
        "held-out",
        "--leak-channel-split",
        "transfer",
        "--leak-channel-splits",
        str(config),
    ]

    default_open = generate.parse_args(common)
    explicit_open = generate.parse_args([*common, "--leak-channels", "open"])
    explicit_loose = generate.parse_args([*common, "--leak-channels", "git_unreachable_loose"])

    assert default_open.leak_channels == ("git_unreachable_packed",)
    assert default_open.leak_channels_open is True
    assert explicit_open.leak_channels == default_open.leak_channels
    assert explicit_open.leak_channels_open is True
    assert explicit_loose.leak_channels == ("git_unreachable_loose",)
    assert explicit_loose.leak_channels_open is False
