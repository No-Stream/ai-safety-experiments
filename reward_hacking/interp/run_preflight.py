"""Prove the code that runs on the box is the code that was smoked, before the expensive stages.

The reward-hacking interpretability run rents a GPU box; the code it runs there must be the code
that was smoke-tested here, and the environment must be able to run it -- both checked BEFORE the
expensive stages. Two rented boxes were burned by failures this gate exists to make impossible:

* a ``uv sync`` error was swallowed by a ``tail`` pipe, so the stage chain started against a venv
  with no torch and died on ``ModuleNotFoundError`` seconds in;
* a driver smoke-tested against the dirty dev working tree was shipped as a stale ``git archive
  HEAD``, so it passed ``max_seq_len=`` to a ``JacobianConfig`` the shipped commit did not have --
  a ``TypeError`` after the model load.

The mechanism has two independent checks, both of which fail LOUD (non-zero exit, every discrepancy
listed at once rather than one at a time):

* a CONTENT MANIFEST pins every shipped code file to a sha256. It is built at ship time over the
  committed ``HEAD`` tree and verified on the box against the extracted tree, so any skew between
  the smoked code and the shipped code -- a stale file, a truncated transfer, an injected extra
  module -- is caught byte-for-byte. This is what turns "smoked == shipped" from a hope into a
  checked fact, and it is the direct fix for the second failure above: a shipped ``jacobian.py``
  that differs from the one the driver was written against cannot pass this check.
* an IMPORT smoke imports every module the run uses on the shipped tree, so a broken
  concurrent-session edit, a syntax error, or a missing gitignored data file (``prompt_contrast``
  and ``eval_awareness_probe`` read baked cases at import) crashes here in milliseconds rather than
  after an 80-minute lens fit.

``build_manifest`` / ``verify_manifest`` / ``default_code_paths`` are pure filesystem logic and are
what the offline tests exercise -- including a sabotage that tampers a shipped file and watches the
verify go red. ``check_run_imports`` runs on the shipped tree; the offline tests drive it with the
data-free interp modules (``jacobian`` / ``generation_capture`` / ``steering`` / ``directions``),
since importing ``prompt_contrast`` needs data present only on the box. ``check_runtime`` -- torch +
CUDA, the ``jlens`` import, the weight cache -- is GPU/box-only and behind a function, mirroring the
rest of the interp package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1

# The whole package, not the ``interp`` + ``harness`` pair this used to name. That pair covered what
# the chain imports directly and so missed ``model_backend.py``, where ``VLLMBackend`` lives: reached
# transitively, newest code in the run, and exactly the file whose staleness this gate exists to
# catch. Naming the package also covers a module added later without anyone widening this by hand.
DEFAULT_CODE_SUBTREES: tuple[str, ...] = ("reward_hacking",)

# Modules imported on the box preflight. The order is load-bearing only in that a failure names the
# first module that would not import. ``run_harness`` is last so its import exercises the whole
# interp graph one more time from the driver's own entry point.
DEFAULT_RUN_MODULES: tuple[str, ...] = (
    "reward_hacking.interp.directions",
    "reward_hacking.interp.jacobian",
    "reward_hacking.interp.generation_capture",
    "reward_hacking.interp.steering",
    "reward_hacking.interp.prompt_contrast",
    "reward_hacking.interp.eval_awareness_probe",
    "reward_hacking.interp.run_harness",
)


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file, read in chunks so a large file is not slurped whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_py_files(root: Path, subtrees: Sequence[str]) -> Iterator[str]:
    """Yield every non-``__pycache__`` ``*.py`` under each subtree, as a POSIX relpath from root.

    Sorted within each subtree so a manifest built twice on the same tree is byte-identical, which
    keeps the manifest diffable and a re-ship idempotent.
    """
    for subtree in subtrees:
        base = root / subtree
        if not base.is_dir():
            raise RuntimeError(
                f"code subtree {subtree!r} is not a directory under {root} -- the manifest would "
                "silently cover nothing"
            )
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path.relative_to(root).as_posix()


def default_code_paths(root: Path, subtrees: Sequence[str] = DEFAULT_CODE_SUBTREES) -> list[str]:
    """Return the sorted list of code files the manifest covers, relative to ``root``."""
    return sorted(_iter_py_files(root, subtrees))


def build_manifest(
    root: Path,
    *,
    git_sha: str,
    subtrees: Sequence[str] = DEFAULT_CODE_SUBTREES,
) -> dict[str, object]:
    """Build a content manifest of the run's code surface, for verification on the box.

    Runs at SHIP time, over the committed ``HEAD`` tree (the caller commits the whole working tree
    first, so ``HEAD`` is the live code). Records the subtrees so the box can re-glob and notice a
    file that exists there but not here (an injected or stale extra module), the ``git_sha`` for the
    marker/log trail, and a sha256 per file.
    """
    files = {rel: sha256_file(root / rel) for rel in default_code_paths(root, subtrees)}
    logger.info("built manifest: %d files under %s at HEAD %s", len(files), subtrees, git_sha)
    return {
        "version": MANIFEST_VERSION,
        "git_sha": git_sha,
        "subtrees": list(subtrees),
        "files": files,
    }


def _manifest_str(manifest: Mapping[str, object], key: str) -> str:
    """Read a required string field, failing loud on a manifest that is the wrong shape."""
    value = manifest.get(key)
    if not isinstance(value, str):
        raise TypeError(f"manifest field {key!r} must be a string, got {type(value).__name__}")
    return value


def _manifest_files(manifest: Mapping[str, object]) -> dict[str, str]:
    """Read the ``files`` mapping, failing loud if it is not a ``{relpath: sha256}`` dict."""
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError(f"manifest 'files' must be an object, got {type(files).__name__}")
    for rel, sha in files.items():
        if not isinstance(rel, str) or not isinstance(sha, str):
            raise TypeError("manifest 'files' must map string paths to string sha256 hex")
    return dict(files)


def _manifest_subtrees(manifest: Mapping[str, object]) -> list[str]:
    """Read the ``subtrees`` list, failing loud if it is not a list of strings."""
    subtrees = manifest.get("subtrees")
    if not isinstance(subtrees, list) or not all(isinstance(item, str) for item in subtrees):
        raise TypeError("manifest 'subtrees' must be a list of strings")
    return list(subtrees)


def verify_manifest(root: Path, manifest: Mapping[str, object]) -> None:
    """Verify the tree under ``root`` matches ``manifest`` exactly; raise listing EVERY discrepancy.

    Three ways the shipped tree can diverge from the smoked one, all fatal and all reported together
    so the operator sees the whole skew rather than fixing one and re-shipping to find the next:

    * ``changed`` -- a file's bytes differ (the stale-``jacobian.py`` failure);
    * ``missing`` -- a manifested file is absent (a truncated or partial transfer);
    * ``extra``   -- a ``*.py`` exists under a covered subtree but is not in the manifest (an
      injected or left-over module that could shadow an import).

    The version guard rejects a manifest written by an incompatible future build rather than
    misreading its shape.
    """
    version = manifest.get("version")
    if version != MANIFEST_VERSION:
        raise RuntimeError(
            f"manifest version {version!r} != supported {MANIFEST_VERSION}; regenerate it"
        )
    git_sha = _manifest_str(manifest, "git_sha")
    expected = _manifest_files(manifest)
    subtrees = _manifest_subtrees(manifest)

    on_disk = set(default_code_paths(root, subtrees))
    problems: list[str] = []
    for rel in sorted(expected):
        path = root / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
            continue
        actual = sha256_file(path)
        if actual != expected[rel]:
            problems.append(f"changed: {rel} (want {expected[rel][:12]}, got {actual[:12]})")
    problems.extend(
        f"extra: {rel} (present on the box, not in the manifest)"
        for rel in sorted(on_disk - set(expected))
    )

    if problems:
        joined = "\n  ".join(problems)
        raise RuntimeError(
            f"shipped code does not match the manifest built from HEAD {git_sha}; "
            f"smoked != shipped, refusing to run:\n  {joined}"
        )
    logger.info("preflight manifest OK: %d files match HEAD %s", len(expected), git_sha)


def check_run_imports(modules: Sequence[str] = DEFAULT_RUN_MODULES) -> None:
    """Import every module the run uses, so an import-time failure surfaces in milliseconds.

    Catches a syntax error a concurrent session left in a shipped module, an API rename that breaks
    an import, and -- because ``prompt_contrast`` / ``eval_awareness_probe`` read baked case data at
    import -- a missing gitignored data file that would otherwise kill the run at its first stage.
    Uses ``importlib`` rather than top-level ``import`` statements so the module list is data, and a
    failure names the offending module rather than aborting this file's own import.
    """
    for name in modules:
        importlib.import_module(name)
    logger.info("preflight imports OK: %d modules import on the shipped tree", len(modules))


def check_runtime(model_id: str | None, *, require_cuda: bool = True) -> None:
    """GPU/box-only: torch imports with a CUDA device, ``jlens`` imports, weights are cached.

    Duplicates the bootstrap's torch gate on purpose (defence in depth): this is the single call
    the runbook makes the box's go/no-go gate, so it must not assume the bootstrap already checked.
    Not exercised by the offline tests -- there is no torch CUDA device and no ``jlens`` on the CI
    box -- so it lives behind a function and is only reached via ``--check-runtime``.
    """
    torch = importlib.import_module("torch")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible to torch; refusing to start the interp run")
    importlib.import_module("jlens")  # raises ModuleNotFoundError if PYTHONPATH is not set
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.info("preflight runtime OK: torch %s on %s", torch.__version__, device)
    if model_id is not None:
        hub = importlib.import_module("huggingface_hub")
        path = hub.snapshot_download(model_id)
        logger.info("preflight weights OK: %s cached at %s", model_id, path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: ``build`` a manifest at ship time, or ``verify`` the shipped tree on the box."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="write a content manifest of the code surface (ship time)")
    build.add_argument("--root", type=Path, default=Path(), help="repo root to hash under")
    build.add_argument("--git-sha", required=True, help="the committed HEAD sha being shipped")
    build.add_argument("--out", type=Path, required=True, help="where to write the manifest json")

    verify = sub.add_parser("verify", help="verify the shipped tree against a manifest (box)")
    verify.add_argument("--root", type=Path, default=Path(), help="extracted repo root on the box")
    verify.add_argument("--manifest", type=Path, required=True, help="the shipped manifest json")
    verify.add_argument(
        "--check-imports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also import the run's modules on the shipped tree (default on)",
    )
    verify.add_argument(
        "--check-runtime",
        action="store_true",
        help="also check torch+CUDA, the jlens import and the weight cache (box only)",
    )
    verify.add_argument(
        "--model-id",
        default=None,
        help="if set with --check-runtime, prefetch/verify these weights are cached",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build or verify the preflight manifest; any failure raises and exits non-zero."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "build":
        manifest = build_manifest(args.root, git_sha=args.git_sha)
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        logger.info("wrote manifest for HEAD %s to %s", args.git_sha, args.out)
        return
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, dict):
        raise TypeError(f"manifest at {args.manifest} is not a JSON object")
    verify_manifest(args.root, manifest)
    if args.check_imports:
        check_run_imports()
    if args.check_runtime:
        check_runtime(args.model_id)
    logger.info("preflight PASSED; safe to start the interp run")


if __name__ == "__main__":
    main()
