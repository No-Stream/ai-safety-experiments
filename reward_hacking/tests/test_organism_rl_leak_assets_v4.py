"""Acceptance tests for selecting prepared leak arms by task and channel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.organism_rl import leak_assets, rollout
from reward_hacking.organism_rl.leak_selection import add_leak_arguments, resolve_leak_options
from reward_hacking.r2e_leak_data import R2eLeakTask

if TYPE_CHECKING:
    from reward_hacking.harness.task_spec import Task

TRAIN_CHANNELS = ("git_branch", "git_remote_ahead", "git_stash")
SINGLE_TRAIN_CHANNELS = (*TRAIN_CHANNELS, "patch_file", "build_cache")


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


@dataclass(frozen=True)
class _EpisodeTask:
    task_id: str
    problem_id: str
    host_only_metadata: dict[str, object]
    docker_config: DockerConfig
    task_variant: str = "leak"
    math_evidence_template: None = None
    grader_timeout_seconds: float | None = None


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
            docker_config=DockerConfig(
                image=f"example.invalid/repository@sha256:{'b' * 64}",
                gate_record=tmp_path / "gate.json",
            ),
        )
        return cast("Task", task)

    monkeypatch.setattr(leak_assets, "build_task", build_task_stub)
    index = {
        "train-task": [
            _asset_arm(tmp_path, "train-task", TRAIN_CHANNELS),
            *(_asset_arm(tmp_path, "train-task", (channel,)) for channel in SINGLE_TRAIN_CHANNELS),
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


def test_train_loader_applies_task_filter_without_pairing_channels(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, built_channels = synthetic_leak_index
    other_train_case = _case("unselected-train-task", "other-seen-repo", "train")
    monkeypatch.setattr(
        leak_assets,
        "load_tasks",
        lambda: [_case("train-task", "seen-repo", "train"), other_train_case],
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index[other_train_case.task_id] = [
        _asset_arm(index_path.parent, other_train_case.task_id, TRAIN_CHANNELS)
    ]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    tasks, _manifests, skips = leak_assets.load_leak_tasks(
        index_path,
        partition="train",
        channels=TRAIN_CHANNELS,
        instruction="none",
        repos=None,
        task_ids=("train-task",),
    )

    assert [task.task_id for task in tasks] == ["train-task"]
    assert built_channels == [TRAIN_CHANNELS]
    assert tasks[0].host_only_metadata["leak_channels_open"] == list(TRAIN_CHANNELS)
    assert not skips


def test_train_loader_selects_reviewed_task_channel_pair_and_opens_only_that_channel(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
) -> None:
    index_path, built_channels = synthetic_leak_index

    tasks, manifests, skips = leak_assets.load_leak_tasks(
        index_path,
        partition="train",
        channels=("build_cache", "patch_file"),
        instruction="none",
        repos=None,
        task_ids=("train-task",),
        task_channel_pairs=(("train-task", "patch_file"),),
    )

    assert [task.task_id for task in tasks] == ["train-task"]
    assert built_channels[-1] == ("patch_file",)
    assert tasks[0].host_only_metadata["leak_channels_open"] == ["patch_file"]
    assert manifests["train-task"]["channels"] == ["patch_file"]
    assert not skips


def test_train_loader_rejects_manifest_that_opens_an_unselected_channel(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
) -> None:
    index_path, _built_channels = synthetic_leak_index
    index = json.loads(index_path.read_text(encoding="utf-8"))
    patch_arm = next(arm for arm in index["train-task"] if arm["channels"] == ["patch_file"])
    manifest_path = Path(patch_arm["manifest_path"])
    manifest_path.write_text(
        json.dumps({"channels": ["build_cache", "patch_file"], "unavailable_channels": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest channels do not partition the requested arm"):
        leak_assets.load_leak_tasks(
            index_path,
            partition="train",
            channels=("build_cache", "patch_file"),
            instruction="none",
            repos=None,
            task_ids=("train-task",),
            task_channel_pairs=(("train-task", "patch_file"),),
        )


def test_train_loader_rejects_task_filter_outside_train_or_unknown(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
) -> None:
    index_path, _built_channels = synthetic_leak_index

    for task_ids, error in (
        (("heldout-seen-task",), "outside the train partition"),
        (("missing-task",), "unknown task ids"),
    ):
        with pytest.raises(ValueError, match=error):
            leak_assets.load_leak_tasks(
                index_path,
                partition="train",
                channels=("patch_file",),
                instruction="none",
                repos=None,
                task_ids=task_ids,
                task_channel_pairs=((task_ids[0], "patch_file"),),
            )


def test_episode_source_forwards_eval_pairs_and_keeps_pair_identity(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_path, built_channels = synthetic_leak_index
    backend = SimpleNamespace(tokenizer=object(), sampling=None)
    monkeypatch.setattr(rollout, "VLLMHTTPBackend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(rollout, "require_gate", lambda _config: None)
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


def test_episode_source_forwards_single_channel_train_pairs(
    synthetic_leak_index: tuple[Path, list[tuple[str, ...]]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_path, built_channels = synthetic_leak_index
    backend = SimpleNamespace(tokenizer=object(), sampling=None)
    monkeypatch.setattr(rollout, "VLLMHTTPBackend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(rollout, "require_gate", lambda _config: None)
    monkeypatch.setattr(
        rollout, "QWEN_PROTOCOL", SimpleNamespace(stop_token_ids=lambda _tokenizer: ())
    )

    source = rollout.TaskEpisodeSource(
        endpoint="http://synthetic:8000",
        model_path="synthetic-model",
        served_model_id="synthetic-model",
        output_dir=tmp_path / "episodes",
        max_model_len=4096,
        families=("r2e_leak",),
        partition="train",
        variants=("oneoff",),
        leak_assets=index_path,
        leak_channels=("build_cache", "patch_file"),
        leak_channels_open=False,
        leak_task_ids=("train-task",),
        leak_train_pairs=(("train-task", "build_cache"),),
    )

    assert tuple(source.tasks) == ("train-task",)
    assert built_channels[-1] == ("build_cache",)
    assert source.tasks["train-task"].host_only_metadata["leak_channels_open"] == ["build_cache"]


def test_leak_task_source_metadata_binds_filter_and_channel_pairs(
    tmp_path: Path,
) -> None:
    asset_index = tmp_path / "assets.json"
    asset_index.write_text("{}", encoding="utf-8")
    task_filter = tmp_path / "train-cases.json"
    filter_bytes = b'{"partition":"train","task_ids":["train-task"]}\n'
    task_filter.write_bytes(filter_bytes)
    filter_digest = hashlib.sha256(filter_bytes).hexdigest()

    metadata = leak_assets.leak_task_source_metadata(
        asset_index,
        partition="train",
        channels=("patch_file",),
        instruction="none",
        repos=None,
        eval_pairs=None,
        task_filter_path=task_filter,
        task_ids=("train-task",),
        train_pairs=(("train-task", "patch_file"),),
    )

    assert metadata["task_filter"] == {
        "path": str(task_filter),
        "sha256": filter_digest,
        "task_ids": ["train-task"],
        "task_channel_pairs": [["train-task", "patch_file"]],
    }
    reformatted_filter = b'{ "task_ids": ["train-task"], "partition": "train" }\n'
    task_filter.write_bytes(reformatted_filter)
    reformatted_metadata = leak_assets.leak_task_source_metadata(
        asset_index,
        partition="train",
        channels=("patch_file",),
        instruction="none",
        repos=None,
        eval_pairs=None,
        task_filter_path=task_filter,
        task_ids=("train-task",),
        train_pairs=(("train-task", "patch_file"),),
    )
    assert (
        cast("dict[str, object]", reformatted_metadata["task_filter"])["sha256"]
        == hashlib.sha256(reformatted_filter).hexdigest()
    )

    task_filter.write_text('{"partition":"train","task_ids":["changed"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="resolved task selection"):
        leak_assets.leak_task_source_metadata(
            asset_index,
            partition="train",
            channels=("patch_file",),
            instruction="none",
            repos=None,
            eval_pairs=None,
            task_filter_path=task_filter,
            task_ids=("train-task",),
            train_pairs=(("train-task", "patch_file"),),
        )


def test_single_channel_train_pair_reaches_persisted_episode_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _EpisodeTask(
        task_id="train-task",
        problem_id="synthetic-repository/revision",
        host_only_metadata={
            "leak_channels_open": ["patch_file"],
            "leak_channels_unavailable": {},
            "leak_repo_seen": True,
            "instruction_variant": "none",
        },
        docker_config=DockerConfig(
            image=f"example.invalid/repository@sha256:{'b' * 64}",
            gate_record=tmp_path / "gate.json",
        ),
    )
    manifest: dict[str, object] = {"channels": ["patch_file"], "unavailable_channels": {}}
    selected: dict[str, object] = {}

    def load_selected_task(
        *_args: object,
        manifest_json: dict[str, str] | None,
        task_ids: tuple[str, ...] | None,
        task_channel_pairs: tuple[tuple[str, str], ...] | None,
        **_kwargs: object,
    ) -> tuple[list[Task], dict[str, dict[str, object]], dict[str, int]]:
        selected["task_ids"] = task_ids
        selected["task_channel_pairs"] = task_channel_pairs
        if manifest_json is not None:
            manifest_json[task.task_id] = json.dumps(manifest)
        return [cast("Task", task)], {task.task_id: manifest}, {}

    backend = SimpleNamespace(
        model_id="synthetic-served-model",
        sampling=rollout.SamplingConfig(max_new_tokens=16),
        tokenizer=object(),
        stop_token_ids=(),
    )
    monkeypatch.setattr(rollout, "load_leak_tasks", load_selected_task)
    monkeypatch.setattr(rollout, "require_gate", lambda _config: None)
    monkeypatch.setattr(rollout, "VLLMHTTPBackend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(
        rollout, "QWEN_PROTOCOL", SimpleNamespace(stop_token_ids=lambda _tokenizer: ())
    )
    monkeypatch.setattr(rollout, "task_definition_sha256", lambda _task: "synthetic-digest")
    monkeypatch.setattr(rollout, "task_definition_content", lambda _task, **_kwargs: {})
    monkeypatch.setattr(
        rollout, "_task_submission_grader", lambda *_args, **_kwargs: lambda _path: None
    )
    monkeypatch.setattr(
        rollout,
        "run_ladder_episode",
        lambda *_args, **_kwargs: SimpleNamespace(
            summary_record=lambda: {
                "record": "episode_summary",
                "true": {"measured": True, "pass": False},
                "true_pass": False,
            }
        ),
    )
    monkeypatch.setattr(rollout, "_finalize_leak_summary", lambda *_args, **_kwargs: None)

    source = rollout.TaskEpisodeSource(
        endpoint="http://synthetic:8000",
        model_path="synthetic-model",
        served_model_id="synthetic-served-model",
        output_dir=tmp_path / "episodes",
        max_model_len=4096,
        families=("r2e_leak",),
        partition="train",
        variants=("oneoff",),
        leak_channels=("build_cache", "patch_file"),
        leak_channels_open=False,
        leak_task_ids=("train-task",),
        leak_train_pairs=(("train-task", "patch_file"),),
    )

    source.payment_registers["train-task"] = "trusted_rerun"
    sample = source("train-task", seed=11, step=2, replica=0)
    identity = next(
        record for record in sample.trace_records if record["record"] == "organism_episode_identity"
    )

    assert selected == {
        "task_ids": ("train-task",),
        "task_channel_pairs": (("train-task", "patch_file"),),
    }
    assert identity["leak_channels_open"] == ["patch_file"]


def test_real_smoke_indexes_load_their_prepared_arms() -> None:
    asset_root = os.environ.get("LEAK_V4_ASSET_ROOT")
    if asset_root is None:
        pytest.skip("set LEAK_V4_ASSET_ROOT to the directory containing v4 leak indexes")
    root = Path(asset_root)
    train_path = root / "assets-v9-train" / "assets.json"
    heldout_path = root / "assets-v8-heldout" / "assets.json"
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
    expected_pairs = json.loads(split_path.read_text(encoding="utf-8"))["eval_pairs"]
    assert len(args.leak_eval_pairs) == len(heldout_tasks) == len(expected_pairs)
    identity_records = [task.host_only_metadata for task in heldout_tasks]
    assert all(
        isinstance(identity["leak_channels_open"], list)
        and len(identity["leak_channels_open"]) == 1
        for identity in identity_records
    )
    assert all(isinstance(identity["leak_repo_seen"], bool) for identity in identity_records)


def _git(repository: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Dev",
        "GIT_AUTHOR_EMAIL": "dev@example.invalid",
        "GIT_COMMITTER_NAME": "Dev",
        "GIT_COMMITTER_EMAIL": "dev@example.invalid",
        "GIT_AUTHOR_DATE": "2020-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2020-01-01T00:00:00Z",
    }
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(repository), *arguments],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()


def _repository_with_planted_fix(root: Path, channel: str) -> tuple[Path, dict[str, object]]:
    repository = root / channel
    repository.mkdir(parents=True)
    _git(repository, "init", "-q", "-b", "main")
    (repository / "module.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    (repository / "unchanged.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", "base")
    base_commit = _git(repository, "rev-parse", "HEAD")
    (repository / "module.py").write_text("def answer():\n    return 2\n", encoding="utf-8")
    (repository / "added.py").write_text("NEW = True\n", encoding="utf-8")
    _git(repository, "add", "-A")
    if channel == "git_branch":
        _git(repository, "commit", "-q", "-m", "fix")
        fix_commit = _git(repository, "rev-parse", "HEAD")
        fix_tree = _git(repository, "rev-parse", "HEAD^{tree}")
        _git(repository, "branch", "reference-fix")
        _git(repository, "reset", "-q", "--hard", base_commit)
    else:
        _git(repository, "stash", "push", "-q", "-m", "wip")
        fix_tree = _git(repository, "rev-parse", "refs/stash^{tree}")
        # The asset builder records the upstream fix commit, which a stash-only arm never contains.
        fix_commit = "f" * 40
    manifest: dict[str, object] = {
        "channels": [channel],
        "unavailable_channels": {},
        "fix": {"base_commit_id": base_commit, "commit_id": fix_commit, "tree_id": fix_tree},
    }
    return repository, manifest


def test_stash_only_arm_records_the_same_reference_files_as_a_branch_arm(tmp_path: Path) -> None:
    contents: dict[str, object] = {}
    for channel in ("git_branch", "git_stash"):
        repository, manifest = _repository_with_planted_fix(tmp_path, channel)
        task = SimpleNamespace(workspace_template=repository, host_only_metadata={})
        content = rollout._channel_content(cast("Task", task), manifest)  # pyright: ignore[reportPrivateUsage]
        contents[channel] = content["reference_files"]
    assert contents["git_stash"] == contents["git_branch"]
    reference_files = cast("dict[str, dict[str, str]]", contents["git_branch"])
    assert sorted(reference_files) == ["added.py", "module.py"]
    assert (
        bytes.fromhex(reference_files["module.py"]["content"]) == b"def answer():\n    return 2\n"
    )
