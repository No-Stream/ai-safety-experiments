"""Acceptance tests for selecting prepared leak arms by task and channel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.organism_rl import leak_assets, rollout
from reward_hacking.organism_rl.leak_selection import add_leak_arguments, resolve_leak_options
from reward_hacking.r2e_leak_data import R2eLeakTask

if TYPE_CHECKING:
    from reward_hacking.harness.task_spec import Task

TRAIN_CHANNELS = ("git_branch", "git_remote_ahead", "git_stash")


def _case(task_id: str, repo_name: str, partition: str) -> R2eLeakTask:
    return R2eLeakTask(
        task_id=task_id,
        repo_name=repo_name,
        commit_hash="a" * 40,
        docker_image="example.invalid/repository:tag",
        image_digest="sha256:" + "b" * 64,
        problem_statement="",
        expected_output={},
        gold_patch="",
        modified_files=(),
        num_non_test_files=0,
        num_non_test_lines=0,
        partition=partition,
    )


def _asset_arm(tmp_path: Path, task_id: str, channels: tuple[str, ...]) -> dict[str, object]:
    arm_dir = tmp_path / task_id / "-".join(channels)
    arm_dir.mkdir(parents=True)
    manifest_path = arm_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"channels": list(channels), "unavailable_channels": {}}),
        encoding="utf-8",
    )
    return {
        "channels": list(channels),
        "extracted_repository": str(tmp_path / task_id / "extracted"),
        "workspace_template": str(arm_dir / "open"),
        "closed_template": str(arm_dir / "closed"),
        "manifest_path": str(manifest_path),
        "gate_record": str(arm_dir / "gate.json"),
        "hidden_tests_template": str(arm_dir / "hidden-tests"),
        "runtime_venv": str(arm_dir / "venv"),
        "runtime_interpreter": str(arm_dir / "python"),
    }


@pytest.fixture
def synthetic_leak_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[tuple[str, ...]]]:
    train_case = _case("train-task", "seen-repo", "train")
    seen_heldout_case = _case("heldout-seen-task", "seen-repo", "held-out-task")
    unseen_heldout_case = _case("heldout-unseen-task", "unseen-repo", "held-out-repo")
    monkeypatch.setattr(
        leak_assets,
        "load_tasks",
        lambda: [train_case, seen_heldout_case, unseen_heldout_case],
    )
    monkeypatch.setattr(leak_assets, "_case_exclusion", lambda _case, _repository: None)
    built_channels: list[tuple[str, ...]] = []

    def build_task_stub(
        case: R2eLeakTask,
        *,
        required_channels: tuple[str, ...],
        **_kwargs: object,
    ) -> Task:
        built_channels.append(required_channels)
        task = SimpleNamespace(
            task_id=case.task_id,
            problem_id=f"{case.repo_name}/{case.commit_hash}",
            host_only_metadata={"leak_channels_open": list(required_channels)},
        )
        return cast("Task", task)

    monkeypatch.setattr(leak_assets, "build_task", build_task_stub)
    index = {
        "train-task": [
            _asset_arm(tmp_path, "train-task", TRAIN_CHANNELS),
            _asset_arm(tmp_path, "train-task", ("git_reflog",)),
        ],
        "heldout-seen-task": [
            _asset_arm(tmp_path, "heldout-seen-task", ("git_reflog",)),
            _asset_arm(tmp_path, "heldout-seen-task", ("mtime",)),
        ],
        "heldout-unseen-task": [_asset_arm(tmp_path, "heldout-unseen-task", ("pyc_cache",))],
    }
    index_path = tmp_path / "assets.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path, built_channels


def test_eval_loader_selects_each_requested_task_channel_arm_and_records_identity(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
) -> None:
    index_path, built_channels = synthetic_leak_index

    tasks, _manifests, skips = leak_assets.load_leak_tasks(
        index_path,
        partition="held-out",
        channels=("git_reflog", "mtime", "pyc_cache"),
        instruction="none",
        repos=None,
        eval_pairs=(
            ("heldout-seen-task", "mtime"),
            ("heldout-unseen-task", "pyc_cache"),
        ),
    )

    assert [task.task_id for task in tasks] == ["heldout-seen-task", "heldout-unseen-task"]
    assert built_channels == [("mtime",), ("pyc_cache",)]
    assert not skips
    identity_by_task = {task.task_id: task.host_only_metadata for task in tasks}
    assert identity_by_task["heldout-seen-task"]["leak_channels_open"] == ["mtime"]
    assert identity_by_task["heldout-seen-task"]["leak_repo_seen"] is True
    assert identity_by_task["heldout-unseen-task"]["leak_channels_open"] == ["pyc_cache"]
    assert identity_by_task["heldout-unseen-task"]["leak_repo_seen"] is False


def test_eval_loader_refuses_a_task_channel_pair_without_an_arm(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
) -> None:
    index_path, _built_channels = synthetic_leak_index

    with pytest.raises(ValueError, match="no asset arm"):
        leak_assets.load_leak_tasks(
            index_path,
            partition="held-out",
            channels=("build_cache",),
            instruction="none",
            repos=None,
            eval_pairs=(("heldout-seen-task", "build_cache"),),
        )


def test_train_loader_keeps_exact_channel_set_matching(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
) -> None:
    index_path, built_channels = synthetic_leak_index

    tasks, _manifests, skips = leak_assets.load_leak_tasks(
        index_path,
        partition="train",
        channels=TRAIN_CHANNELS,
        instruction="none",
        repos=None,
    )

    assert [task.task_id for task in tasks] == ["train-task"]
    assert built_channels == [TRAIN_CHANNELS]
    assert not skips


def test_episode_source_forwards_eval_pairs_and_keeps_pair_identity(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_path, built_channels = synthetic_leak_index
    backend = SimpleNamespace(tokenizer=object(), sampling=None)
    monkeypatch.setattr(rollout, "VLLMHTTPBackend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(
        rollout, "QWEN_PROTOCOL", SimpleNamespace(stop_token_ids=lambda _tokenizer: ())
    )
    eval_pairs = (
        ("heldout-seen-task", "mtime"),
        ("heldout-unseen-task", "pyc_cache"),
    )

    source = rollout.TaskEpisodeSource(
        endpoint="http://synthetic:8000",
        model_path="synthetic-model",
        served_model_id="synthetic-model",
        output_dir=tmp_path / "episodes",
        max_model_len=4096,
        families=("r2e_leak",),
        partition="held-out",
        variants=("oneoff",),
        leak_assets=index_path,
        leak_channels=("mtime", "pyc_cache"),
        leak_eval_pairs=eval_pairs,
    )

    assert tuple(source.tasks) == tuple(task_id for task_id, _channel in eval_pairs)
    assert built_channels[-2:] == [("mtime",), ("pyc_cache",)]
    for task_id, channel in eval_pairs:
        identity = source.tasks[task_id].host_only_metadata
        assert identity["leak_channels_open"] == [channel]
        assert isinstance(identity["leak_repo_seen"], bool)


def test_real_smoke_indexes_load_their_prepared_arms() -> None:
    asset_root = os.environ.get("LEAK_V4_ASSET_ROOT")
    if asset_root is None:
        pytest.skip("set LEAK_V4_ASSET_ROOT to the directory containing v4 leak indexes")
    root = Path(asset_root)
    train_path = root / "assets-v6-train" / "assets.json"
    heldout_path = root / "assets-v6-heldout" / "assets.json"
    split_path = Path(
        os.environ.get(
            "LEAK_V4_CHANNEL_SPLITS",
            Path(__file__).resolve().parents[2] / "docs/scratch/leak-v4/channel-splits.json",
        )
    )
    if not train_path.is_file() or not heldout_path.is_file() or not split_path.is_file():
        pytest.skip("prepared v4 leak indexes or channel split are absent")

    train_index = json.loads(train_path.read_text(encoding="utf-8"))
    train_task_id, train_arms = next(iter(train_index.items()))
    train_channels = tuple(train_arms[0]["channels"])
    train_tasks, _train_manifests, _train_skips = leak_assets.load_leak_tasks(
        train_path,
        partition="train",
        channels=train_channels,
        instruction="none",
        repos=None,
    )
    assert train_task_id in {task.task_id for task in train_tasks}

    parser = argparse.ArgumentParser()
    parser.add_argument("--family", dest="family_entries", action="append")
    add_leak_arguments(parser)
    args = parser.parse_args(
        [
            "--family",
            "r2e_leak",
            "--leak-assets",
            str(heldout_path),
            "--leak-channel-split",
            "seed-v4",
            "--leak-channel-splits",
            str(split_path),
            "--leak-channel-per-episode",
        ]
    )
    resolve_leak_options(args, parser, partition="held-out")
    heldout_tasks, _heldout_manifests, _heldout_skips = leak_assets.load_leak_tasks(
        heldout_path,
        partition="held-out",
        channels=args.leak_channels,
        instruction=args.leak_instruction,
        repos=args.leak_repos,
        eval_pairs=args.leak_eval_pairs,
    )
    assert len(args.leak_eval_pairs) == len(heldout_tasks) == 8
    identity_records = [task.host_only_metadata for task in heldout_tasks]
    assert all(
        isinstance(identity["leak_channels_open"], list)
        and len(identity["leak_channels_open"]) == 1
        for identity in identity_records
    )
    assert all(isinstance(identity["leak_repo_seen"], bool) for identity in identity_records)
