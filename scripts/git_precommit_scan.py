"""Refuse a ``git commit`` whose STAGED blobs carry text the privacy rule bans.

``make privacy-scan`` and ``tests/test_scan_secrets.py`` sweep files on DISK, and disk is
the wrong surface at commit time: on 2026-08-22 pre-strip survey files sat staged in the
shared index while every file on disk scanned clean, one session's ``git add <one-file> &&
git commit`` swept the whole index, and three local commits carried instrument item text
before anything looked. This script closes that seam. It scans the exact blobs the commit
would record -- ``git show :<path>`` per staged path, under whatever index git handed the
hook -- so nothing that reaches history goes unscanned, however it got into the index.

Two hook modes, both installed as thin wrappers by ``scripts/install_git_hooks.sh``:

  (no flag)          pre-commit: scan every staged blob, refuse the commit on any finding.
  --message-file F   commit-msg: scan the commit message draft. Messages have carried
                     registered benchmark values before, and no other gate reads them.

Detectors, arming, and the finding format (path, line, detector, fingerprint -- never the
matched text) all come from ``scripts/scan_secrets.py``; this file only points them at the
index. Arming reads the MAIN worktree's local item sources, so a commit made from a linked
agent worktree (which carries no gitignored files) is scanned with the same corpus as one
made from the main tree. A missing canary token file arms nothing and is deliberately quiet:
the canary values protected a former shared box and the owner no longer keeps them. Without
item sources -- a fresh clone -- the instrument detector is INERT and says so loudly, but the
commit is not refused for it: a fresh clone has no contraband to leak, and the shape
detectors still run.

Staging one of the arming source files ITSELF is refused outright rather than scanned:
the sources are the one place item text is allowed to live, and they are exempt from
their own findings inside the scanner, so without this rule they would be the one thing
that could ride a commit through unflagged.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import scan_secrets

logger = logging.getLogger(__name__)

CANARY_FILE_RELATIVE = Path("canary") / "privacy-values.txt"


def _git_lines(*args: str) -> str:
    """Run one git command and return its stdout, crashing loudly if git refuses.

    The environment is inherited on purpose: git points hooks at the commit's real index
    through GIT_INDEX_FILE (a ``git commit <pathspec>`` builds a temporary one), and
    scrubbing the environment would silently scan the wrong index.
    """
    result = subprocess.run(  # noqa: S603 - fixed git binary, arguments built from literals
        ("git", *args),  # noqa: S607 - resolved from PATH on purpose: hooks must use the caller's git
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8", "replace")


def main_worktree_root() -> Path:
    """Resolve the main worktree's root, which is where the gitignored arming sources live.

    ``--git-common-dir`` rather than ``--show-toplevel`` because hooks also fire for
    commits made inside linked agent worktrees, where the toplevel is a checkout that has
    no gitignored files at all -- arming from there would silently disarm the one detector
    this hook exists for.
    """
    common_dir = Path(_git_lines("rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    return common_dir.parent


def staged_paths() -> list[str]:
    """Every path the pending commit would write, deletions excluded."""
    raw = _git_lines("diff", "--cached", "--name-only", "--diff-filter=d", "-z")
    return [path for path in raw.split("\0") if path]


def staged_blob_text(path: str) -> str:
    """Read one path's STAGED content -- the bytes the commit would record, not the file on disk."""
    return _git_lines("show", f":{path}")


def _is_arming_source(path: str) -> bool:
    """Whether a staged path is one of the local files the instrument detector arms from."""
    posix = Path(path).as_posix()
    in_data_dir = posix.startswith(scan_secrets.INSTRUMENT_DATA_DIR + "/") and posix.endswith(
        ".json"
    )
    return in_data_dir or posix == scan_secrets.INSTRUMENT_RETRIEVAL_NOTE


def _scan_one_text(
    label: str,
    text: str,
    canaries: tuple[scan_secrets.CanaryToken, ...],
    sources: scan_secrets.InstrumentSources,
) -> list[scan_secrets.Finding]:
    """Run every detector family scan_secrets has over one piece of text.

    The label keeps the path's suffix (so the whole-file benchmark check still knows a
    .json from a .md) but is prefixed with git's ``:<path>`` index notation, which also
    guarantees it can never resolve to an arming source's on-disk path and inherit that
    file's self-exemption.
    """
    findings = scan_secrets.scan_text(label, text, canaries)
    findings.extend(scan_secrets.benchmark_item_file_findings(label, text))
    findings.extend(scan_secrets.instrument_text_findings(label, text, sources))
    return findings


def _load_arming(
    repo_root: Path,
) -> tuple[scan_secrets.InstrumentSources, tuple[scan_secrets.CanaryToken, ...]]:
    """Arm the instrument and canary detectors from the main worktree's local files."""
    sources = scan_secrets.collect_instrument_sources(repo_root)
    canary_path = repo_root / CANARY_FILE_RELATIVE
    canaries = scan_secrets.load_canary_tokens(str(canary_path) if canary_path.is_file() else None)
    if not sources.is_armed:
        logger.warning(
            "instrument-text detector is INERT: no local item sources under %s. On a fresh "
            "clone that is expected and safe (nothing local to leak); on the research box it "
            "means this gate is NOT protecting instrument text -- stop and investigate.",
            repo_root,
        )
    logger.info(
        f"armed: {len(scan_secrets.DETECTORS)} shape detectors, "
        f"{len(sources.phrases)} instrument phrase(s), {len(sources.payoff_items)} payoff "
        f"item(s), {len(canaries)} canary token(s)"
    )
    return sources, canaries


def scan_staged_blobs(repo_root: Path) -> int:
    """Scan every staged blob; non-zero means the commit must not happen."""
    sources, canaries = _load_arming(repo_root)
    paths = staged_paths()
    findings: list[scan_secrets.Finding] = []
    for path in paths:
        if _is_arming_source(path):
            findings.append(
                scan_secrets.Finding(f":{path}", 1, "instrument_source_file_staged", "-")
            )
            continue
        blob = staged_blob_text(path)
        if len(blob) > scan_secrets.DEFAULT_MAX_BYTES:
            logger.error(
                f":{path} is {len(blob)} bytes staged, over the {scan_secrets.DEFAULT_MAX_BYTES} "
                "scan limit. Refusing rather than skipping it silently."
            )
            return 1
        findings.extend(_scan_one_text(f":{path}", blob, canaries, sources))
    if findings:
        return _report(findings, f"{len(paths)} staged blob(s)")
    logger.info(f"clean: {len(paths)} staged blob(s)")
    return 0


def scan_message_file(repo_root: Path, message_path: str) -> int:
    """Scan a commit message draft, which no file-oriented gate ever reads."""
    sources, canaries = _load_arming(repo_root)
    text = Path(message_path).read_text(encoding="utf-8", errors="replace")
    findings = _scan_one_text("commit-message", text, canaries, sources)
    if findings:
        return _report(findings, "the commit message")
    logger.info("clean: commit message")
    return 0


def _report(findings: list[scan_secrets.Finding], scanned: str) -> int:
    """Name every finding (never its text) and refuse the commit."""
    for finding in findings:
        logger.error(str(finding))
    detectors = sorted({finding.detector for finding in findings})
    logger.error(
        f"REFUSING the commit: {len(findings)} finding(s) in {scanned}, detectors {detectors}. "
        "The staged content -- not necessarily the file on disk -- carries banned text; fix the "
        "file, re-add it, and retry. Findings name detectors and lines, never the matched text."
    )
    return 1


def main() -> int:
    """Entry point for both hook wrappers."""
    parser = argparse.ArgumentParser(
        description="Refuse a commit whose staged blobs (or message) carry banned text."
    )
    parser.add_argument(
        "--message-file",
        help="scan this commit message draft instead of the staged blobs (commit-msg mode)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="git-privacy-hook: %(message)s", stream=sys.stderr
    )
    repo_root = main_worktree_root()
    if args.message_file is not None:
        return scan_message_file(repo_root, args.message_file)
    return scan_staged_blobs(repo_root)


if __name__ == "__main__":
    sys.exit(main())
