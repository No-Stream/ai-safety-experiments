"""Fetch a chosen TMAX artifact from the Hugging Face hub, or (``--dry-run``) print the plan.

Every downloadable is a named :class:`DownloadTarget` derived from the verified registry in
:mod:`reward_hacking.tmax.artifacts`, so resolving one that was never verified fails loudly before
any bytes move. The flagship model target carries an ``ignore_patterns`` of ``rollouts/*`` because
``allenai/tmax-9b``'s ``main`` branch bundles a ~44 GB (compressed) ``rollouts/`` folder alongside
the weights; the ``rollouts`` target is the mirror image, pulling only the transcripts and manifests.
The size ladder's other rungs are ``model-tmax_2b`` / ``model-tmax_4b`` / ``model-tmax_27b`` with
their RL-init bases as ``base-tmax_2b`` / ``base-tmax_4b`` / ``base-tmax_27b``; the 9B rung is the
existing ``model-tmax_15k`` and ``base`` pair.

``--dry-run`` never touches the network: it logs exactly what would be fetched and returns, which is
what ``reward_hacking/tests/test_tmax_download.py`` exercises with the hub call stubbed out. A real
fetch shells out to ``huggingface_hub.snapshot_download``, into a subdirectory of ``--local-dir``
named for the artifact and its revision -- the seven suite checkpoints are all Qwen3.5-9B, and the
dose-response ladder is one repo at five revisions, so both would otherwise blend.

**Redirects and stubs.** A hub ``/resolve/<revision>/<file>`` URL answers with a redirect to the
CDN, so a hand-rolled ``curl`` without ``-L`` saves the redirect body as if it were the file and
reports success. This module never shells to curl: ``snapshot_download`` follows that redirect itself
(``hf_hub_download`` reads the ``Location`` of its metadata request and downloads from there) and
raises its own consistency error when the landed byte count differs from the hub's advertised size.
Because a fetch that returns is still not a fetch that landed the weights, :func:`download_artifact`
then runs :func:`verify_snapshot_payload` over what landed: no empty payload file, no file that
begins with an HTTP redirect body, every safetensors file with a well-formed header, and -- where
the registry knows the checkpoint's byte total -- the safetensors summing to exactly that.

``--verify-rung NAME`` runs the registry's size-ladder claim against the live hub without
downloading anything: every branch the registry lists exists and no other does, ``main`` carries the
same safetensors object ids as the step branch it is said to duplicate, and every step branch is
distinct weights (:func:`reward_hacking.tmax.artifacts.reconcile_branch_weight_ids`).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFile

from reward_hacking.tmax.artifacts import (
    BASE_CHECKPOINT,
    DEFAULT_REVISION,
    FLAGSHIP_ROLLOUTS,
    SIZE_LADDER,
    SUITES,
    TMAX_9B,
    UPSTREAM_BASE,
    UPSTREAM_INSTRUCT,
    Checkpoint,
    RolloutArchive,
    SizeRung,
    reconcile_branch_weight_ids,
    size_rung,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from huggingface_hub.hf_api import GitRefs, RepoFolder

logger = logging.getLogger(__name__)

# The weight/config/tokenizer files a checkpoint snapshot needs, without any rollouts/ payload.
WEIGHTS_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.json",
    "*.jinja",
    "tokenizer*",
    "*.model",
)

# The flagship main branch bundles rollouts/ with the weights; exclude it from a weights snapshot.
WEIGHTS_IGNORE_PATTERNS: tuple[str, ...] = ("rollouts/*",)

# A safetensors file is a little-endian u64 header length, the JSON header, then the tensor bytes.
SAFETENSORS_HEADER_LENGTH_BYTES = 8
STUB_SNIFF_BYTES = 64
# How an unfollowed hub redirect, or an HTML error page saved as a file, begins.
REDIRECT_STUB_SIGNATURES: tuple[bytes, ...] = (
    b"<!DOCTYPE",
    b"<html",
    b"<HTML",
    b"Temporary Redirect",
    b"Moved Permanently",
    b"Found. Redirecting",
)
# The file kinds a snapshot here can legitimately contain; anything else is skipped, not judged.
PAYLOAD_SUFFIXES: tuple[str, ...] = (
    ".safetensors",
    ".json",
    ".jinja",
    ".jsonl",
    ".zst",
    ".md",
    ".csv",
    ".parquet",
    ".model",
)
# snapshot_download keeps its own bookkeeping (empty .lock files, .metadata) under this directory.
HUB_METADATA_DIRNAME = ".cache"


class SnapshotIntegrityError(RuntimeError):
    """Raised when a returned snapshot does not hold the payload its file names promise."""


@dataclass(frozen=True)
class DownloadTarget:
    """One named thing to snapshot: a repo, its type, the revision, and the pattern filters.

    ``expected_safetensors_bytes`` is the registry's byte total for the checkpoint's weights, when
    it knows one; :func:`verify_snapshot_payload` holds the landed safetensors to exactly it.
    """

    name: str
    repo_id: str
    repo_type: str
    revision: str
    allow_patterns: tuple[str, ...] | None
    ignore_patterns: tuple[str, ...] | None
    description: str
    expected_safetensors_bytes: int | None = None


def _model_target(name: str, checkpoint: Checkpoint, description: str) -> DownloadTarget:
    """Build a weights-only model snapshot (no rollouts), resolving the checkpoint fail-loud."""
    repo_id, revision = checkpoint.resolve()
    return DownloadTarget(
        name=name,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=WEIGHTS_ALLOW_PATTERNS,
        ignore_patterns=WEIGHTS_IGNORE_PATTERNS,
        description=description,
        expected_safetensors_bytes=checkpoint.safetensors_bytes,
    )


def _rollouts_target(name: str, archive: RolloutArchive) -> DownloadTarget:
    """Build the training-rollout transcripts snapshot (skips the ~226 GB of logprobs)."""
    repo_id, revision = archive.resolve()
    return DownloadTarget(
        name=name,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=archive.rollouts_only_patterns,
        ignore_patterns=None,
        description="Flagship training rollouts (transcripts + manifests only).",
    )


def download_targets() -> dict[str, DownloadTarget]:
    """Build every named downloadable from the registry: bases, models, datasets, rollouts.

    Suite model targets are ``model-<suite>``; dataset targets ``dataset-<suite>``; the non-9B size
    rungs are ``model-<rung>`` with ``base-<rung>`` for their RL-init bases. The flagship model is
    fetched at ``--revision main`` by default; pass ``--revision step_100`` etc. for a step branch.
    """
    targets: dict[str, DownloadTarget] = {
        "base": _model_target(
            "base", BASE_CHECKPOINT, "The matched RL-init base checkpoint (hamishivi/Qwen3.5-9B)."
        ),
        "base-upstream-instruct": _model_target(
            "base-upstream-instruct", UPSTREAM_INSTRUCT, "Upstream instruct base (Qwen/Qwen3.5-9B)."
        ),
        "base-upstream-base": _model_target(
            "base-upstream-base", UPSTREAM_BASE, "Upstream -Base (interp SAE/lens are fit here)."
        ),
        "rollouts": _rollouts_target("rollouts", FLAGSHIP_ROLLOUTS),
    }
    for name, entry in SUITES.items():
        targets[f"model-{name}"] = _model_target(
            f"model-{name}", entry.rl_model, f"RL'd Qwen3.5-9B for the {entry.display_name} suite."
        )
        targets[f"dataset-{name}"] = DownloadTarget(
            name=f"dataset-{name}",
            repo_id=entry.rl_dataset.resolve(),
            repo_type="dataset",
            revision="main",
            allow_patterns=None,
            ignore_patterns=None,
            description=f"RL environment data for the {entry.display_name} suite.",
        )
    for rung in SIZE_LADDER.values():
        if rung is TMAX_9B:
            continue  # already present as model-tmax_15k and base
        targets[f"model-{rung.name}"] = _model_target(
            f"model-{rung.name}",
            rung.checkpoint(),
            f"{rung.params_label} flagship-recipe checkpoint ({rung.repo_id}); "
            f"main == {rung.main_branch}.",
        )
        targets[f"base-{rung.name}"] = _model_target(
            f"base-{rung.name}",
            rung.base(),
            f"RL-init base for the {rung.params_label} rung ({rung.base_mirror_repo_id}).",
        )
    return targets


def _artifact_local_dir(target: DownloadTarget, local_dir: Path | None) -> Path | None:
    """Namespace a ``--local-dir`` by target name *and* revision; the hub cache stays ``None``.

    The seven suite checkpoints are all Qwen3.5-9B, so their shard filenames and
    ``model.safetensors.index.json`` collide. Two of them fetched into one directory would leave a
    blended checkpoint that ``from_pretrained`` loads without complaint and no result table can
    reveal; a subdirectory removes that mode rather than detecting it.

    The revision is half of that, because the target name alone does not identify a checkpoint:
    ``dose_response_ladder`` is one repo at five revisions, so ``--artifact model-tmax_15k
    --revision step_100`` and ``--revision step_500`` are the same name and different weights. The
    ladder is also the measurement least able to survive the blend, since it is looking for a
    monotone curve across exactly those rungs.

    Spelled the way :attr:`~reward_hacking.tmax.artifacts.Checkpoint.model_ref` spells it: the
    default revision is implicit, so only a pin is written out, and directories already fetched at
    ``main`` keep their paths. Path separators in a revision are folded, so a ``refs/pr/N`` ref
    stays one directory rather than becoming a tree.
    """
    if local_dir is None:
        return None
    if target.revision == DEFAULT_REVISION:
        return local_dir / target.name
    return local_dir / f"{target.name}@{target.revision.replace('/', '-')}"


def _is_payload(path: Path) -> bool:
    """Whether a landed file is one the snapshot is expected to carry (and so gets judged)."""
    return path.suffix in PAYLOAD_SUFFIXES or ".part-" in path.name


def _check_safetensors_header(path: Path, head: bytes, size: int) -> None:
    """Refuse a safetensors file whose first bytes are not a header the format could produce."""
    if len(head) <= SAFETENSORS_HEADER_LENGTH_BYTES:
        raise SnapshotIntegrityError(f"{path} is {size} bytes, shorter than a safetensors header")
    header_length = int.from_bytes(head[:SAFETENSORS_HEADER_LENGTH_BYTES], "little")
    opens_json = head[SAFETENSORS_HEADER_LENGTH_BYTES : SAFETENSORS_HEADER_LENGTH_BYTES + 1] == b"{"
    if not opens_json or SAFETENSORS_HEADER_LENGTH_BYTES + header_length > size:
        raise SnapshotIntegrityError(
            f"{path} does not begin with a safetensors header (declared header length "
            f"{header_length}, file size {size}, opens JSON={opens_json}); not a weights file"
        )


def verify_snapshot_payload(
    snapshot_dir: Path, *, expected_safetensors_bytes: int | None = None
) -> None:
    """Fail loudly if what landed under ``snapshot_dir`` is not the payload its names promise.

    Judged per payload file (:data:`PAYLOAD_SUFFIXES` and ``.part-NNN`` pieces; the hub's own
    ``.cache`` bookkeeping is skipped): not empty, not beginning with an HTTP redirect body, and for
    safetensors a well-formed header. With ``expected_safetensors_bytes`` the safetensors must also
    sum to exactly that, which is the check a redirect stub, a partial write, or a pattern that
    silently matched nothing cannot pass.
    """
    if not snapshot_dir.is_dir():
        raise SnapshotIntegrityError(f"{snapshot_dir} is not a directory; nothing landed there")
    safetensors_total = 0
    judged = 0
    for path in sorted(snapshot_dir.rglob("*")):
        relative_parts = path.relative_to(snapshot_dir).parts
        if HUB_METADATA_DIRNAME in relative_parts or not path.is_file() or not _is_payload(path):
            continue
        judged += 1
        size = path.stat().st_size
        if size == 0:
            raise SnapshotIntegrityError(
                f"{path} is empty: a redirect stub or an interrupted write, not the file it names"
            )
        with path.open("rb") as handle:
            head = handle.read(STUB_SNIFF_BYTES)
        if head.startswith(REDIRECT_STUB_SIGNATURES):
            raise SnapshotIntegrityError(
                f"{path} begins with an HTTP redirect body ({head!r}); the hub's /resolve/ "
                f"URL was saved instead of followed"
            )
        if path.suffix == ".safetensors":
            _check_safetensors_header(path, head, size)
            safetensors_total += size
    if judged == 0:
        raise SnapshotIntegrityError(f"no payload files landed under {snapshot_dir}")
    if expected_safetensors_bytes is not None and safetensors_total != expected_safetensors_bytes:
        raise SnapshotIntegrityError(
            f"safetensors under {snapshot_dir} total {safetensors_total} bytes; the registry "
            f"expects {expected_safetensors_bytes} for this checkpoint"
        )


def download_artifact(
    target: DownloadTarget,
    *,
    cache_dir: Path | None = None,
    local_dir: Path | None = None,
) -> Path:
    """Snapshot one target from the hub, verify what landed, and return the local path."""
    destination = _artifact_local_dir(target, local_dir)
    logger.info(
        "downloading %s: %s (%s@%s) allow=%s ignore=%s into=%s",
        target.name,
        target.repo_id,
        target.repo_type,
        target.revision,
        target.allow_patterns,
        target.ignore_patterns,
        destination or "hub cache",
    )
    path = Path(
        snapshot_download(
            repo_id=target.repo_id,
            repo_type=target.repo_type,
            revision=target.revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_dir=None if destination is None else str(destination),
            allow_patterns=list(target.allow_patterns) if target.allow_patterns else None,
            ignore_patterns=list(target.ignore_patterns) if target.ignore_patterns else None,
        )
    )
    verify_snapshot_payload(path, expected_safetensors_bytes=target.expected_safetensors_bytes)
    return path


class RepoListing(Protocol):
    """The two ``HfApi`` calls the branch check reads; a test double supplies the same two."""

    def list_repo_refs(self, repo_id: str) -> GitRefs:
        """Return the repo's git refs (branches with names and target commits)."""
        ...

    def list_repo_tree(
        self, repo_id: str, *, revision: str, recursive: bool
    ) -> Iterable[RepoFile | RepoFolder]:
        """Yield the repo's files and folders at ``revision``."""
        ...


def hub_branch_weight_ids(repo_id: str, api: RepoListing) -> dict[str, tuple[str, ...]]:
    """List ``{branch: sorted safetensors LFS sha256 ids}`` for every branch of a repo.

    The object ids are what "main == step_200" means on the hub: two branches whose safetensors
    resolve to the same LFS objects are the same weights whatever their commits say. A safetensors
    entry that is not an LFS object raises, because weights are never stored inline.
    """
    ids: dict[str, tuple[str, ...]] = {}
    for branch in api.list_repo_refs(repo_id).branches:
        objects: list[str] = []
        for entry in api.list_repo_tree(repo_id, revision=branch.name, recursive=True):
            if not isinstance(entry, RepoFile) or not entry.path.endswith(".safetensors"):
                continue
            if entry.lfs is None:
                raise SnapshotIntegrityError(
                    f"{repo_id}@{branch.name}: {entry.path} is not an LFS object; not weights"
                )
            objects.append(entry.lfs.sha256)
        ids[branch.name] = tuple(sorted(objects))
    return ids


def verify_rung_release(rung: SizeRung, api: RepoListing) -> dict[str, tuple[str, ...]]:
    """Check one size rung's registry claim against the live hub; return the listing that passed."""
    ids = hub_branch_weight_ids(rung.repo_id, api)
    reconcile_branch_weight_ids(rung, ids)
    return ids


def _log_plan(target: DownloadTarget, *, cache_dir: Path | None, local_dir: Path | None) -> None:
    """Log exactly what a fetch of ``target`` would pull -- the ``--dry-run`` output, no network."""
    logger.info("[dry-run] %s -- %s", target.name, target.description)
    logger.info(
        "    repo_id=%s repo_type=%s revision=%s", target.repo_id, target.repo_type, target.revision
    )
    logger.info("    allow_patterns=%s", target.allow_patterns)
    logger.info("    ignore_patterns=%s", target.ignore_patterns)
    logger.info("    expected_safetensors_bytes=%s", target.expected_safetensors_bytes)
    logger.info("    cache_dir=%s local_dir=%s", cache_dir, _artifact_local_dir(target, local_dir))


def main(argv: list[str] | None = None, *, hub_api: RepoListing | None = None) -> int:
    """Resolve the chosen artifact and either download it or (``--dry-run``) print the plan.

    ``hub_api`` is only for ``--verify-rung``; tests pass a listing double, the CLI a real
    :class:`~huggingface_hub.HfApi`.
    """
    targets = download_targets()
    parser = argparse.ArgumentParser(
        description="Download a TMAX Stage-A artifact from Hugging Face."
    )
    parser.add_argument(
        "--artifact", choices=sorted(targets), help="Which named artifact to fetch."
    )
    parser.add_argument(
        "--revision", default=None, help="Override the git revision (e.g. a flagship step branch)."
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Hugging Face cache directory."
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Download into a per-artifact subdirectory of this directory instead of the cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be fetched without any network access.",
    )
    parser.add_argument("--list", action="store_true", help="List available artifacts and exit.")
    parser.add_argument(
        "--verify-rung",
        choices=sorted(SIZE_LADDER),
        default=None,
        help="Check a size rung's branches and main==step mapping against the live hub, no download.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.verify_rung is not None:
        rung = size_rung(args.verify_rung)
        ids = verify_rung_release(rung, hub_api if hub_api is not None else HfApi())
        for branch, objects in sorted(ids.items()):
            logger.info("%s@%s safetensors ids: %s", rung.repo_id, branch, objects)
        logger.info(
            "%s: %d branches as registered; main == %s; step branches distinct",
            rung.repo_id,
            len(ids),
            rung.main_branch,
        )
        return 0

    if args.list or args.artifact is None:
        logger.info("available artifacts:")
        for name in sorted(targets):
            logger.info("  %-26s %s", name, targets[name].description)
        if args.artifact is None and not args.list:
            parser.error("pass --artifact NAME (or --list to see the choices)")
        return 0

    target = targets[args.artifact]
    if args.revision is not None:
        # A field-by-field copy silently drops fields added later, e.g. the rollouts ignore list.
        target = replace(target, revision=args.revision)

    if args.dry_run:
        _log_plan(target, cache_dir=args.cache_dir, local_dir=args.local_dir)
        return 0

    path = download_artifact(target, cache_dir=args.cache_dir, local_dir=args.local_dir)
    logger.info("downloaded %s to %s", target.name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
