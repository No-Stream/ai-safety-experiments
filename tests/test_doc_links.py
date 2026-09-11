"""Every relative markdown link in a tracked doc must resolve in a fresh clone.

The failure this catches is silent and has nearly happened twice: a doc is reorganised, links are
repointed at the new location, and either the new files are never `git add`ed or the target sits
under a gitignored path. Locally everything resolves because the files are on disk, so nothing looks
wrong; a fresh clone gets an index full of dead links. Checking against `git ls-files` rather than
the filesystem is what makes the difference — `docs/scratch/` is deliberately untracked, so a link
into it resolves here and nowhere else.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GIT = "/usr/bin/git"

# Matches the target of a markdown inline link, i.e. the parenthesised part of [text](target).
MARKDOWN_LINK_TARGET = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:")


def tracked_paths() -> set[str]:
    completed = subprocess.run(  # noqa: S603 - trusted executable and literal arguments
        [GIT, "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(completed.stdout.split())


def resolves(target: str, tracked: set[str]) -> bool:
    if target in tracked:
        return True
    # A link may point at a directory, which is tracked only through the files inside it.
    as_dir = target.rstrip("/") + "/"
    return any(path.startswith(as_dir) for path in tracked)


def test_relative_markdown_links_resolve_in_a_fresh_clone() -> None:
    tracked = tracked_paths()
    markdown_files = sorted(path for path in tracked if path.endswith(".md"))
    assert markdown_files, "expected tracked markdown files; git ls-files returned none"

    dangling: list[str] = []
    checked = 0
    for doc in markdown_files:
        containing_dir = (REPO_ROOT / doc).parent
        text = (REPO_ROOT / doc).read_text(encoding="utf-8", errors="replace")
        for match in MARKDOWN_LINK_TARGET.finditer(text):
            target = match.group(1).split("#")[0].strip()
            if not target or target.startswith(EXTERNAL_PREFIXES):
                continue
            checked += 1
            # resolve() collapses `..` segments so a link can be compared against git's paths.
            absolute = (containing_dir / target).resolve()
            if not absolute.is_relative_to(REPO_ROOT):
                dangling.append(f"{doc} -> {target} (points outside the repository)")
                continue
            relative = absolute.relative_to(REPO_ROOT).as_posix()
            if not resolves(relative, tracked):
                dangling.append(f"{doc} -> {target} (resolves to {relative}, not tracked)")

    assert checked, "expected to find relative links to check; the link regex may have broken"
    report = "\n  ".join(dangling)
    assert not dangling, f"dangling relative links, dead in a fresh clone:\n  {report}"
