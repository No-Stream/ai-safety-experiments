"""Exercise the TMAX download CLI offline: the filters, where bytes land, and what counts as landed.

The module docstring promises ``--dry-run`` never touches the network and says that is what the
offline tests exercise. This is those tests. ``huggingface_hub.snapshot_download`` is replaced with
a recorder throughout, so nothing here can reach the hub even by accident, and the assertions are on
the arguments a real fetch would be handed.

Three of them are worth more than the rest. ``rollouts/*`` in every weights target's ignore list is
the only thing keeping a snapshot of ``allenai/tmax-9b`` from also dragging down its ~44 GB
``rollouts/`` folder, and a dropped or misspelled pattern would be invisible until a fetch had
already run for hours. A ``--local-dir`` has to be namespaced per artifact *and* per revision:
experiment 2 fetches seven same-shaped Qwen3.5-9B checkpoints, experiment 1's ladder fetches one
repo at five revisions, and in both cases the shard filenames and ``model.safetensors.index.json``
collide, so two of them sharing one directory would silently produce a blended checkpoint that
``from_pretrained`` loads happily and no downstream table can reveal. And a fetch that returns is
not a fetch that landed the weights: the payload check is fed a redirect stub, an empty file, a
mangled safetensors header and a wrong byte total, each of which it has to refuse, before the
consumer is shown to call it and to propagate its refusal.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from huggingface_hub.hf_api import GitRefInfo, GitRefs, RepoFile, RepoFolder

# Runtime-public but absent from utils' __all__; it is the function snapshot_download filters with.
from huggingface_hub.utils import filter_repo_objects  # pyright: ignore[reportPrivateImportUsage]

from reward_hacking.tmax import download
from reward_hacking.tmax.artifacts import (
    FLAGSHIP_ROLLOUTS,
    SIZE_LADDER,
    SUITES,
    TMAX_2B,
    TMAX_4B,
    TMAX_9B_BYTES,
    TMAX_27B,
    SizeRung,
    TmaxArtifactError,
)
from reward_hacking.tmax.download import (
    WEIGHTS_ALLOW_PATTERNS,
    WEIGHTS_IGNORE_PATTERNS,
    SnapshotIntegrityError,
    download_artifact,
    download_targets,
    hub_branch_weight_ids,
    main,
    verify_rung_release,
    verify_snapshot_payload,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

ROLLOUTS_TARGET = "rollouts"
ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"


def write_tiny_safetensors(path: Path) -> int:
    """Write the smallest well-formed safetensors file (one F32 scalar); return its byte size."""
    header = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    payload = len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload)


class RecordingSnapshotDownload:
    """Stands in for ``snapshot_download``: records the kwargs, lands a tiny valid snapshot."""

    def __init__(self, hub_cache: Path) -> None:
        self.calls: list[dict[str, object]] = []
        self.hub_cache = hub_cache

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        local_dir = kwargs["local_dir"]
        if local_dir is None:
            landed = self.hub_cache / str(kwargs["repo_id"]).replace("/", "--")
        else:
            landed = Path(str(local_dir))
        write_tiny_safetensors(landed / "model.safetensors")
        return str(landed)


class RecordingVerifier:
    """Stands in for ``verify_snapshot_payload``: records what the consumer asked it to judge."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, int | None]] = []

    def __call__(self, snapshot_dir: Path, *, expected_safetensors_bytes: int | None) -> None:
        self.calls.append((snapshot_dir, expected_safetensors_bytes))


def refuse_to_fetch(**kwargs: object) -> str:
    """A ``snapshot_download`` that fails the test if a supposedly offline path calls it."""
    raise AssertionError(f"snapshot_download must not be called here, got {sorted(kwargs)}")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RecordingSnapshotDownload:
    """Swap the hub call for a recorder and the payload check for its own recorder.

    The plumbing tests below are about kwargs and paths; the payload check has its own tests
    against the real files it judges, and ``TestTheConsumerRunsThePayloadCheck`` shows the two
    are wired together.
    """
    stub = RecordingSnapshotDownload(tmp_path / "hub-cache")
    monkeypatch.setattr(download, "snapshot_download", stub)
    monkeypatch.setattr(download, "verify_snapshot_payload", RecordingVerifier())
    return stub


def weights_targets() -> dict[str, download.DownloadTarget]:
    """Every model-weights target: three 9B bases, seven suite models, three rungs and their bases."""
    return {
        name: target
        for name, target in download_targets().items()
        if target.repo_type == "model" and name != ROLLOUTS_TARGET
    }


class TestTheWeightsTargetsExcludeTheRolloutsFolder:
    """The ~44 GB guard: it has to be attached to every weights target and forwarded to the hub."""

    def test_there_are_sixteen_weights_targets_and_every_one_ignores_rollouts(self) -> None:
        targets = weights_targets()
        assert len(targets) == 3 + len(SUITES) + 2 * (len(SIZE_LADDER) - 1)
        for name, target in targets.items():
            assert target.ignore_patterns is not None, name
            assert "rollouts/*" in target.ignore_patterns, name

    def test_every_weights_target_allows_safetensors_shards(self) -> None:
        for name, target in weights_targets().items():
            assert target.allow_patterns is not None, name
            assert "*.safetensors" in target.allow_patterns, name
            assert target.allow_patterns == WEIGHTS_ALLOW_PATTERNS, name

    def test_the_ignore_list_matches_the_registry_s_own_mirror_pattern(self) -> None:
        """The registry and the CLI must not drift apart on which folder to skip."""
        assert FLAGSHIP_ROLLOUTS.weights_ignore_patterns == WEIGHTS_IGNORE_PATTERNS

    def test_the_filters_reach_the_hub_call_as_lists(
        self, recorder: RecordingSnapshotDownload
    ) -> None:
        download_artifact(download_targets()["model-tmax_15k"])
        (call,) = recorder.calls
        assert call["ignore_patterns"] == ["rollouts/*"]
        assert call["allow_patterns"] == list(WEIGHTS_ALLOW_PATTERNS)
        assert call["repo_id"] == "allenai/tmax-9b"
        assert call["repo_type"] == "model"


class TestTheSizeLadderTargets:
    """One weights target per non-9B rung and one per RL-init base, each with its byte total."""

    def test_the_rung_targets_name_their_repos_and_expected_bytes(self) -> None:
        targets = download_targets()
        two_b = targets["model-tmax_2b"]
        assert (two_b.repo_id, two_b.revision) == ("allenai/tmax-2b", "main")
        assert two_b.expected_safetensors_bytes == TMAX_2B.weight_layout.bytes_per_branch
        base_27b = targets["base-tmax_27b"]
        assert base_27b.repo_id == "hamishivi/Qwen3.6-27B"
        assert base_27b.expected_safetensors_bytes == TMAX_27B.base_safetensors_bytes
        assert "main == step_160" in targets["model-tmax_27b"].description

    def test_the_9b_rung_is_not_duplicated_under_a_second_name(self) -> None:
        targets = download_targets()
        assert "model-tmax_9b" not in targets
        assert "base-tmax_9b" not in targets
        assert targets["model-tmax_15k"].expected_safetensors_bytes == TMAX_9B_BYTES
        assert (
            targets["base"].expected_safetensors_bytes
            == SIZE_LADDER["tmax_9b"].base_safetensors_bytes
        )

    def test_every_9b_suite_model_expects_the_one_9b_byte_total(self) -> None:
        for name in SUITES:
            assert download_targets()[f"model-{name}"].expected_safetensors_bytes == TMAX_9B_BYTES

    def test_the_unmeasured_upstream_base_expects_nothing(self) -> None:
        """No byte total was read for Qwen3.5-9B-Base, so the check must not invent one."""
        assert download_targets()["base-upstream-base"].expected_safetensors_bytes is None

    def test_a_revision_override_keeps_the_byte_total(self) -> None:
        """Every branch of a rung has the same byte total, so the pin travels with the override."""
        pinned = replace(download_targets()["model-tmax_2b"], revision="step_300")
        assert pinned.expected_safetensors_bytes == TMAX_2B.weight_layout.bytes_per_branch


class TestTheRolloutsTargetIsTheMirrorImage:
    """The other half of the split: transcripts only, and no weights."""

    def test_it_allows_only_rollout_paths_and_ignores_nothing(self) -> None:
        target = download_targets()[ROLLOUTS_TARGET]
        assert target.ignore_patterns is None
        assert target.allow_patterns == FLAGSHIP_ROLLOUTS.rollouts_only_patterns
        assert target.allow_patterns is not None
        assert all(pattern.startswith("rollouts/") for pattern in target.allow_patterns)
        assert "*.safetensors" not in target.allow_patterns
        assert target.expected_safetensors_bytes is None

    def test_it_skips_the_logprobs_that_dominate_the_archive(self) -> None:
        """The ~226 GB of logprobs are excluded by the allow list being NARROWER than ``rollouts/``.

        Not "by the allow list being rollouts-only" (this docstring's wording until 2026-08-24,
        matching the ``rollouts_only_patterns`` field name): the logprobs live INSIDE the same
        ``rollouts/`` folder, so rollouts-only is precisely the reading under which they would come
        along. What excludes them is that the patterns select the nested transcript paths
        (``rollouts/archives/*/rollouts/*``) plus manifests, and nothing else.

        Know what this test does NOT pin: the assertion below is lexical -- no pattern contains the
        substring "logprob" -- and would still pass if someone widened the list to ``rollouts/*``,
        which pulls the 226 GB. It stays because it catches a different mistake (a pattern written
        FOR the logprobs); the exclusion itself is pinned by the semantic test below, against real
        hub paths and the same matcher the download uses.
        """
        target = download_targets()[ROLLOUTS_TARGET]
        assert target.allow_patterns is not None
        assert not any("logprob" in pattern for pattern in target.allow_patterns)

    def test_a_real_logprob_path_is_not_selected_by_the_matcher_the_download_uses(self) -> None:
        """The semantic half: the exclusion itself, pinned where the lexical test above cannot.

        The three paths are real, copied verbatim from the hub listing of ``allenai/tmax-9b``
        (``list_repo_files``, 2026-08-24, re-listed 2026-09-02: 8 logprob parts across 7 fragments,
        all under ``rollouts/archives/*/training_logprobs/``). ``filter_repo_objects`` is the
        function ``snapshot_download`` itself filters with, so this asserts against the consumption
        point rather than against pattern text. The two positive controls are load-bearing: without
        them, a broken matcher that selects nothing would pass the exclusion assertion vacuously.

        Sabotage contrast, watched 2026-08-24: widening the allow list to ``rollouts/*`` reddens
        THIS test while the lexical test above stays green -- substring absence was never evidence
        of exclusion, which is why both tests exist.
        """
        target = download_targets()[ROLLOUTS_TARGET]
        assert target.allow_patterns is not None
        archive = "swerl_qwen35_9b_fp32lm_dppo_g32__42__1779647982"
        logprob = (
            f"rollouts/archives/{archive}/training_logprobs/training_logprobs.tar.zst.part-000"
        )
        transcript = f"rollouts/archives/{archive}/rollouts/rollouts.tar.zst.part-000"
        manifest = f"rollouts/manifests/{archive}.jsonl"
        selected = set(
            filter_repo_objects(
                [logprob, transcript, manifest], allow_patterns=list(target.allow_patterns)
            )
        )
        assert transcript in selected, "positive control: the matcher must select the transcripts"
        assert manifest in selected, "positive control: the matcher must select the manifests"
        assert logprob not in selected


class TestTheDatasetTargets:
    """One dataset per suite, fetched whole: no pattern filters, and typed as a dataset repo."""

    def test_every_suite_has_an_unfiltered_dataset_target(self) -> None:
        targets = download_targets()
        for name, entry in SUITES.items():
            target = targets[f"dataset-{name}"]
            assert target.repo_type == "dataset"
            assert target.allow_patterns is None
            assert target.ignore_patterns is None
            assert target.repo_id == entry.rl_dataset.resolve()


class TestALocalDirIsNamespacedPerArtifact:
    """Two checkpoints must never be able to blend into one directory."""

    def test_two_targets_sharing_one_local_dir_land_in_separate_subdirectories(
        self, recorder: RecordingSnapshotDownload, tmp_path: Path
    ) -> None:
        targets = download_targets()
        first = download_artifact(targets["model-cli_gym"], local_dir=tmp_path)
        second = download_artifact(targets["model-swe_smith"], local_dir=tmp_path)
        assert first != second
        assert (first.name, second.name) == ("model-cli_gym", "model-swe_smith")
        assert [call["local_dir"] for call in recorder.calls] == [
            str(tmp_path / "model-cli_gym"),
            str(tmp_path / "model-swe_smith"),
        ]

    def test_the_returned_path_is_the_namespaced_one(
        self, recorder: RecordingSnapshotDownload, tmp_path: Path
    ) -> None:
        """A caller handing the return value to ``from_pretrained`` must get the subdirectory."""
        landed = download_artifact(download_targets()["base"], local_dir=tmp_path)
        assert landed == tmp_path / "base"
        assert recorder.calls[0]["local_dir"] == str(tmp_path / "base")

    def test_without_a_local_dir_the_hub_cache_is_used_untouched(
        self, recorder: RecordingSnapshotDownload, tmp_path: Path
    ) -> None:
        download_artifact(download_targets()["base"], cache_dir=tmp_path)
        (call,) = recorder.calls
        assert call["local_dir"] is None
        assert call["cache_dir"] == str(tmp_path)


class TestALocalDirIsNamespacedPerRevisionToo:
    """The dose-response ladder is one repo at five revisions, so the name alone cannot separate it.

    ``dose_response_ladder`` is ``allenai/tmax-9b`` at ``step_100`` through ``step_500``: same repo,
    same target name, same shard filenames, five different sets of weights. Namespacing by target
    name only, two rungs fetched into one ``--local-dir`` blend into a checkpoint that is neither,
    which is the exact failure the per-artifact subdirectory exists to remove -- and the ladder is
    the measurement most able to hide it, since a monotone dose curve is what it hopes to see.
    """

    def test_two_revisions_of_one_target_land_in_separate_subdirectories(
        self, recorder: RecordingSnapshotDownload, tmp_path: Path
    ) -> None:
        flagship = download_targets()["model-tmax_15k"]
        first = download_artifact(replace(flagship, revision="step_100"), local_dir=tmp_path)
        second = download_artifact(replace(flagship, revision="step_500"), local_dir=tmp_path)
        assert first != second, "two ladder rungs would blend into one directory"
        assert [call["local_dir"] for call in recorder.calls] == [str(first), str(second)]

    def test_the_default_revision_keeps_the_bare_target_name(
        self, recorder: RecordingSnapshotDownload, tmp_path: Path
    ) -> None:
        """Follows ``Checkpoint.model_ref``: ``main`` is implicit, so only a pin is spelled out."""
        landed = download_artifact(download_targets()["base"], local_dir=tmp_path)
        assert landed == tmp_path / "base"
        assert recorder.calls[0]["local_dir"] == str(tmp_path / "base")

    def test_a_revision_override_from_the_cli_reaches_the_destination(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """The CLI is the only way to reach a step branch, and it rebuilds the target by hand."""
        monkeypatch.setattr(download, "snapshot_download", refuse_to_fetch)
        with caplog.at_level(logging.INFO, logger="reward_hacking.tmax.download"):
            assert (
                main(
                    [
                        "--artifact",
                        "model-tmax_15k",
                        "--revision",
                        "step_300",
                        "--local-dir",
                        str(tmp_path),
                        "--dry-run",
                    ]
                )
                == 0
            )
        assert f"local_dir={tmp_path / 'model-tmax_15k@step_300'}" in caplog.text


class TestThePayloadCheckRefusesWhatIsNotTheWeights:
    """Each refusal is fed the exact thing it exists to catch; the valid case is the control."""

    def test_a_well_formed_snapshot_passes_with_and_without_a_byte_total(
        self, tmp_path: Path
    ) -> None:
        size = write_tiny_safetensors(tmp_path / "model.safetensors")
        (tmp_path / "config.json").write_text("{}")
        verify_snapshot_payload(tmp_path)
        verify_snapshot_payload(tmp_path, expected_safetensors_bytes=size)

    def test_a_redirect_body_saved_as_the_weights_is_refused(self, tmp_path: Path) -> None:
        """What ``curl`` without ``-L`` leaves behind for a hub /resolve/ URL."""
        (tmp_path / "model.safetensors").write_bytes(
            b"<html>\n<head><title>302 Found</title></head>\n<body>Temporary Redirect</body>"
        )
        with pytest.raises(SnapshotIntegrityError, match="HTTP redirect body"):
            verify_snapshot_payload(tmp_path)

    def test_a_plain_text_redirect_body_is_refused_too(self, tmp_path: Path) -> None:
        (tmp_path / "rollouts.tar.zst.part-000").write_bytes(b"Temporary Redirect. Redirecting to")
        with pytest.raises(SnapshotIntegrityError, match="HTTP redirect body"):
            verify_snapshot_payload(tmp_path)

    def test_an_empty_payload_file_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.json").write_bytes(b"")
        with pytest.raises(SnapshotIntegrityError, match="is empty"):
            verify_snapshot_payload(tmp_path)

    def test_the_hub_s_own_empty_lock_files_are_not_judged(self, tmp_path: Path) -> None:
        """``snapshot_download`` leaves zero-byte ``.lock`` files under ``.cache``; not payload."""
        write_tiny_safetensors(tmp_path / "model.safetensors")
        lock = tmp_path / ".cache" / "huggingface" / "download" / "model.safetensors.lock"
        lock.parent.mkdir(parents=True)
        lock.write_bytes(b"")
        (lock.parent / "model.safetensors.metadata").write_text("etag\n")
        verify_snapshot_payload(tmp_path)

    def test_a_safetensors_file_with_a_mangled_header_is_refused(self, tmp_path: Path) -> None:
        """Right suffix, plausible size, but the header length points past the file's end."""
        bogus = (10**9).to_bytes(8, "little") + b'{"w":{}}' + b"\x00" * 32
        (tmp_path / "model.safetensors").write_bytes(bogus)
        with pytest.raises(
            SnapshotIntegrityError, match="does not begin with a safetensors header"
        ):
            verify_snapshot_payload(tmp_path)

    def test_a_safetensors_file_too_short_for_a_header_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"\x01\x02\x03")
        with pytest.raises(SnapshotIntegrityError, match="shorter than a safetensors header"):
            verify_snapshot_payload(tmp_path)

    def test_a_byte_total_that_misses_the_registry_s_is_refused_naming_both(
        self, tmp_path: Path
    ) -> None:
        size = write_tiny_safetensors(tmp_path / "model.safetensors")
        with pytest.raises(SnapshotIntegrityError, match=f"total {size} bytes.*expects {size + 1}"):
            verify_snapshot_payload(tmp_path, expected_safetensors_bytes=size + 1)

    def test_two_shards_are_summed_against_the_total(self, tmp_path: Path) -> None:
        """The 27B is two files; the check is on their sum, not on any one of them."""
        first = write_tiny_safetensors(tmp_path / "model-00001-of-00002.safetensors")
        second = write_tiny_safetensors(tmp_path / "model-00002-of-00002.safetensors")
        verify_snapshot_payload(tmp_path, expected_safetensors_bytes=first + second)
        with pytest.raises(SnapshotIntegrityError, match="expects"):
            verify_snapshot_payload(tmp_path, expected_safetensors_bytes=first)

    def test_a_snapshot_with_no_payload_at_all_is_refused(self, tmp_path: Path) -> None:
        """A pattern that matched nothing returns an empty directory, which is not a success."""
        (tmp_path / ".gitattributes").write_text("*.safetensors filter=lfs\n")
        with pytest.raises(SnapshotIntegrityError, match="no payload files landed"):
            verify_snapshot_payload(tmp_path)

    def test_a_path_that_is_not_a_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotIntegrityError, match="is not a directory"):
            verify_snapshot_payload(tmp_path / "never-created")

    def test_a_zstd_part_and_a_manifest_pass_as_rollout_payload(self, tmp_path: Path) -> None:
        archive = tmp_path / "rollouts" / "archives" / "fragment" / "rollouts"
        archive.mkdir(parents=True)
        (archive / "rollouts.tar.zst.part-000").write_bytes(ZSTD_FRAME_MAGIC + b"\x00" * 16)
        (tmp_path / "rollouts" / "manifests").mkdir()
        (tmp_path / "rollouts" / "manifests" / "summary.json").write_text('{"total_files": 8424}')
        verify_snapshot_payload(tmp_path)


class TestTheConsumerRunsThePayloadCheck:
    """``download_artifact`` has to call the check on what landed and let its refusal through."""

    def test_the_check_is_handed_the_landed_path_and_the_target_s_byte_total(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fetch = RecordingSnapshotDownload(tmp_path / "hub-cache")
        verifier = RecordingVerifier()
        monkeypatch.setattr(download, "snapshot_download", fetch)
        monkeypatch.setattr(download, "verify_snapshot_payload", verifier)
        landed = download_artifact(download_targets()["model-tmax_2b"], local_dir=tmp_path)
        assert verifier.calls == [(landed, TMAX_2B.weight_layout.bytes_per_branch)]

    def test_end_to_end_a_matching_snapshot_returns_and_a_stub_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The real check, through the real consumer, in both of its states."""
        fetch = RecordingSnapshotDownload(tmp_path / "hub-cache")
        monkeypatch.setattr(download, "snapshot_download", fetch)
        tiny = write_tiny_safetensors(tmp_path / "probe.safetensors")
        target = replace(download_targets()["model-tmax_2b"], expected_safetensors_bytes=tiny)
        assert download_artifact(target, local_dir=tmp_path) == tmp_path / "model-tmax_2b"

        def land_a_stub(**kwargs: object) -> str:
            stub_dir = Path(str(kwargs["local_dir"]))
            stub_dir.mkdir(parents=True, exist_ok=True)
            (stub_dir / "model.safetensors").write_bytes(b"<html>302</html>")
            return str(stub_dir)

        monkeypatch.setattr(download, "snapshot_download", land_a_stub)
        with pytest.raises(SnapshotIntegrityError, match="HTTP redirect body"):
            download_artifact(target, local_dir=tmp_path / "again")

    def test_the_real_registry_total_refuses_a_tiny_snapshot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A rung's true byte total against the recorder's tiny file: the size check bites."""
        monkeypatch.setattr(
            download, "snapshot_download", RecordingSnapshotDownload(tmp_path / "hub-cache")
        )
        with pytest.raises(SnapshotIntegrityError, match="expects 3763692048"):
            download_artifact(download_targets()["model-tmax_2b"], local_dir=tmp_path)


def _lfs_file(path: str, sha256: str) -> RepoFile:
    """A tree entry shaped like the hub's for an LFS-stored safetensors file."""
    return RepoFile(path=path, size=1, oid="blob", lfs={"oid": sha256, "size": 1, "pointerSize": 1})


class FakeHubListing:
    """The two ``HfApi`` calls the branch check reads, answered from a dict of branches."""

    def __init__(self, repo_id: str, trees: dict[str, list[RepoFile | RepoFolder]]) -> None:
        self.repo_id = repo_id
        self.trees = trees
        self.tree_calls: list[tuple[str, str, bool]] = []

    def list_repo_refs(self, repo_id: str) -> GitRefs:
        assert repo_id == self.repo_id
        branches = [
            GitRefInfo(name=name, ref=f"refs/heads/{name}", target_commit=f"commit-{name}")
            for name in self.trees
        ]
        return GitRefs(branches=branches, converts=[], tags=[], pull_requests=None)

    def list_repo_tree(
        self, repo_id: str, *, revision: str, recursive: bool
    ) -> Iterable[RepoFile | RepoFolder]:
        assert repo_id == self.repo_id
        self.tree_calls.append((repo_id, revision, recursive))
        return list(self.trees[revision])


def listing_for(rung: SizeRung, *, main_points_at: str | None = None) -> FakeHubListing:
    """A hub listing in the verified shape for ``rung``, optionally re-pointing ``main``."""
    trees: dict[str, list[RepoFile | RepoFolder]] = {}
    for branch in rung.step_branches:
        entries: list[RepoFile | RepoFolder] = [
            _lfs_file(f"model-{index}.safetensors", f"sha-{branch}-{index}")
            for index in range(rung.weight_layout.safetensors_files)
        ]
        entries.append(RepoFile(path="config.json", size=1, oid="cfg"))
        trees[branch] = entries
    main_source = main_points_at or rung.main_branch
    trees["main"] = [*trees[main_source], RepoFolder(path="rollouts", oid="tree", tree_id="t")]
    return FakeHubListing(rung.repo_id, trees)


class TestVerifyingARungAgainstTheHubListing:
    """The live check, driven by a listing double: no network, the API's own record types."""

    def test_a_listing_in_the_verified_shape_passes_and_returns_the_ids(self) -> None:
        listing = listing_for(TMAX_27B)
        ids = verify_rung_release(TMAX_27B, listing)
        assert set(ids) == {"main", *TMAX_27B.step_branches}
        assert ids["main"] == ids["step_160"] == ("sha-step_160-0", "sha-step_160-1")
        assert all(recursive for _, _, recursive in listing.tree_calls)

    def test_folders_and_non_safetensors_files_are_ignored(self) -> None:
        ids = hub_branch_weight_ids(TMAX_2B.repo_id, listing_for(TMAX_2B))
        assert ids["main"] == ("sha-step_100-0",)

    def test_main_re_pointed_at_another_step_is_refused(self) -> None:
        with pytest.raises(TmaxArtifactError, match="registry says main == step_100"):
            verify_rung_release(TMAX_2B, listing_for(TMAX_2B, main_points_at="step_300"))

    def test_a_safetensors_entry_that_is_not_an_lfs_object_is_refused(self) -> None:
        listing = listing_for(TMAX_2B)
        inline: list[RepoFile | RepoFolder] = [
            RepoFile(path="model.safetensors", size=1, oid="inline")
        ]
        listing.trees["step_200"] = inline
        with pytest.raises(SnapshotIntegrityError, match="not an LFS object"):
            hub_branch_weight_ids(TMAX_2B.repo_id, listing)

    def test_the_cli_flag_runs_the_check_without_fetching_anything(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(download, "snapshot_download", refuse_to_fetch)
        with caplog.at_level(logging.INFO, logger="reward_hacking.tmax.download"):
            assert main(["--verify-rung", "tmax_4b"], hub_api=listing_for(TMAX_4B)) == 0
        assert "allenai/tmax-4b: 5 branches as registered; main == step_200" in caplog.text

    def test_the_cli_flag_fails_loudly_on_a_re_pointed_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(download, "snapshot_download", refuse_to_fetch)
        with pytest.raises(TmaxArtifactError, match="registry says main == step_200"):
            main(
                ["--verify-rung", "tmax_4b"],
                hub_api=listing_for(TMAX_4B, main_points_at="step_380"),
            )


class TestTheDryRunNeverTouchesTheNetwork:
    """``--dry-run`` logs the plan and returns; the hub call is wired to fail if it is reached."""

    def test_it_returns_zero_without_calling_the_hub(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(download, "snapshot_download", refuse_to_fetch)
        with caplog.at_level(logging.INFO, logger="reward_hacking.tmax.download"):
            assert main(["--artifact", "base", "--dry-run"]) == 0
        assert "hamishivi/Qwen3.5-9B" in caplog.text
        assert "rollouts/*" in caplog.text
        assert "expected_safetensors_bytes=19306310880" in caplog.text

    def test_the_plan_names_the_per_artifact_subdirectory_it_would_write(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """A plan naming the parent directory would understate where the bytes actually land."""
        monkeypatch.setattr(download, "snapshot_download", refuse_to_fetch)
        with caplog.at_level(logging.INFO, logger="reward_hacking.tmax.download"):
            assert main(["--artifact", "base", "--local-dir", str(tmp_path), "--dry-run"]) == 0
        assert str(tmp_path / "base") in caplog.text

    def test_a_revision_override_reaches_the_plan(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(download, "snapshot_download", refuse_to_fetch)
        with caplog.at_level(logging.INFO, logger="reward_hacking.tmax.download"):
            assert (
                main(["--artifact", "model-tmax_15k", "--revision", "step_400", "--dry-run"]) == 0
            )
        assert "revision=step_400" in caplog.text

    def test_listing_the_artifacts_is_offline_too(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(download, "snapshot_download", refuse_to_fetch)
        with caplog.at_level(logging.INFO, logger="reward_hacking.tmax.download"):
            assert main(["--list"]) == 0
        assert "model-tmax_15k" in caplog.text
        assert "model-tmax_27b" in caplog.text
        assert "base-tmax_2b" in caplog.text
        assert ROLLOUTS_TARGET in caplog.text

    def test_no_artifact_and_no_list_is_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(download, "snapshot_download", refuse_to_fetch)
        with pytest.raises(SystemExit):
            main([])
