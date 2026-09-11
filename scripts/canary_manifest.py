"""Tripwire over the host's code-execution surfaces.

This exists for the failure we should actually expect: not a jail escape we reasoned
about, but a jail misconfiguration nobody noticed. Reasoning finds the attacks you
thought of; a checksum finds the one you didn't.

The watched set is every path on this box where a write buys arbitrary code execution
the next time something routine happens:

  - ~/.claude/hooks/*             executed by every Claude Code tool call
  - ~/.claude/hooks/lib/*         sourced BY those hooks, so a write here runs on the
                                  same tool call; the parent glob does not descend
  - ~/.claude/settings.json       registers those hooks BY PATH, so repointing an entry
                                  here is code execution without touching a hook file
                                  at all -- the surface a hooks-directory-only tripwire
                                  would miss
  - what settings.json registers  a hook path may point anywhere, and several on this box
                                  point outside ~/.claude; so may the status line and the
                                  credential-export helper. Collected from the file rather
                                  than listed here, so registering a hook watches it
  - the user crontab's targets    run hourly and daily
  - ~/.local/bin/oom-guard.sh     run continuously by an enabled systemd user unit
  - ~/.config/systemd/user/*      those units themselves
  - shell and tool rc files       sourced on every shell; .gitconfig's core.pager and
                                  aliases are execution too

One live surface is deliberately absent: ~/.claude.json, which holds the MCP server
commands. A running session rewrites that file every few seconds, so hashing it would
make this gate permanently red, which is worse than not watching it.

Read-only by construction: it hashes and compares, and never writes to a watched path.
File contents are never printed -- only paths, hashes and a changed/added/removed verdict.

Usage:
  python3 scripts/canary_manifest.py --update     # record the current state as baseline
  python3 scripts/canary_manifest.py              # report drift, exit 1 if any
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = str(Path(__file__).resolve().parent.parent / "canary" / "host-manifest.json")

HOME = str(Path.home())

# Globbed so a new hook is an addition, not a gap; lib/ needs its own, Path.glob not descending.
WATCHED_GLOBS = (
    ".claude/hooks/*",
    ".claude/hooks/lib/*",
    ".config/systemd/user/*.service",
)

SETTINGS_FILE = ".claude/settings.json"

# Keys whose value is a command Claude Code runs; the documented set, not just this box's.
SETTINGS_COMMAND_KEYS = frozenset(
    {"command", "apiKeyHelper", "awsAuthRefresh", "awsCredentialExport"}
)

# A leading redirect operator, optionally file-descriptor-qualified: >, >>, 2>, <, <<, >&.
REDIRECT_PREFIX = re.compile(r"^\d*(?:>>|>&|>|<<|<)")

# Shell punctuation after which the next token is a command rather than an argument.
COMMAND_SEPARATORS = ("{", "}", ";", "|", "&&", "&")

WATCHED_FILES = (
    ".claude/settings.json",
    ".zshrc",
    ".zshenv",
    ".zprofile",
    ".bashrc",
    ".profile",
    ".gitconfig",
    ".condarc",
    ".local/bin/oom-guard.sh",
)

CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Drift:
    """Describe tripwire changes without exposing watched file contents."""

    added: list[str]
    removed: list[str]
    changed: list[str]

    @property
    def clean(self) -> bool:
        """Report whether the watched set is unchanged."""
        return not (self.added or self.removed or self.changed)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_file_targets(command: str) -> set[str]:
    """Absolute paths a command line runs, excluding the files it merely redirects output into.

    The exclusion is the fix for a live bug. Any existing ``/``-prefixed token used to count, so a
    crontab line's ``>> /path/to/log`` put an append-only log in the baseline; cron rewrote the log,
    the hash moved, and ``make canary-check`` went red on a schedule rather than on a compromise. A
    required gate that fails daily for a benign reason is one people learn to wave through.

    Redirects are identified by the operator rather than by testing the executable bit, because
    ``python3 /path/to/job.py`` is code execution on cron's schedule whether or not the file is
    executable, and filtering on the bit would drop that real surface while fixing the log.
    """
    targets = set()
    separated = command
    for separator in COMMAND_SEPARATORS:
        separated = separated.replace(separator, " ")
    next_token_is_redirect_target = False
    for raw in separated.split():
        token = raw.strip("\"'")
        operator = REDIRECT_PREFIX.match(token)
        if operator:
            # A bare operator redirects the following token; an attached one carries its own target.
            next_token_is_redirect_target = operator.end() == len(token)
            continue
        if next_token_is_redirect_target:
            next_token_is_redirect_target = False
            continue
        if token.startswith("~/"):
            token = HOME + token[1:]
        if token.startswith("/") and Path(token).is_file():
            targets.add(token)
    return targets


def _settings_command_strings(node: object) -> list[str]:
    """Every value in a parsed settings tree stored under a key that names a command to run."""
    commands = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SETTINGS_COMMAND_KEYS and isinstance(value, str):
                commands.append(value)
            else:
                commands.extend(_settings_command_strings(value))
    elif isinstance(node, list):
        for item in node:
            commands.extend(_settings_command_strings(item))
    return commands


def settings_command_targets() -> list[str]:
    """Absolute paths the Claude Code settings file registers as commands to run.

    Hashing ``settings.json`` catches an entry being repointed; it does nothing about the file the
    entry points *at*, and a hook may live anywhere -- several on this box sit outside
    ``~/.claude``, where the hook glob cannot see them. Read from settings for the reason cron's
    targets are read from the crontab: registering a hook then watches it, with nothing for
    anybody to remember. A missing settings file is a legitimate state; a malformed one raises out
    of ``json.load``, which is right, because a watched set that silently shrank is the failure.
    """
    settings = Path(HOME) / SETTINGS_FILE
    if not settings.is_file():
        logger.info(f"no Claude Code settings at {settings}; nothing to watch from it")
        return []
    with settings.open() as fh:
        registered = _settings_command_strings(json.load(fh))
    targets = set()
    for command in registered:
        targets.update(command_file_targets(command))
    return sorted(targets)


def crontab_script_targets() -> list[str]:
    """Absolute paths the user crontab invokes, so cron's targets are watched too.

    A missing crontab is a legitimate state (nothing scheduled), but a crontab we cannot
    read is not -- that would silently shrink the watched set, so it raises.
    """
    result = subprocess.run(
        ["crontab", "-l"],  # noqa: S607 - trusted literal command
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if "no crontab" in (result.stderr + result.stdout).lower():
            logger.info("no user crontab installed; nothing to watch from cron")
            return []
        raise RuntimeError(
            f"could not read crontab (rc={result.returncode}): {result.stderr.strip()}"
        )

    targets = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        targets.update(command_file_targets(stripped))
    return sorted(targets)


def watched_paths() -> list[str]:
    """Collect every execution-surface path that the tripwire must hash."""
    paths = set()
    for pattern in WATCHED_GLOBS:
        # Path.glob keeps dotfiles that glob.glob drops from a trailing wildcard; exclude to match.
        paths.update(
            str(p) for p in Path(HOME).glob(pattern) if p.is_file() and not p.name.startswith(".")
        )
    for rel in WATCHED_FILES:
        candidate = str(Path(HOME) / rel)
        if Path(candidate).is_file():
            paths.add(candidate)
    paths.update(settings_command_targets())
    paths.update(crontab_script_targets())
    return sorted(paths)


def build_manifest() -> dict[str, str]:
    """Hash all watched paths into a baseline-ready mapping."""
    return {path: _sha256(path) for path in watched_paths()}


def compare(baseline: dict[str, str], current: dict[str, str]) -> Drift:
    """Identify drift without exposing watched file contents."""
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))
    changed = sorted(p for p in set(baseline) & set(current) if baseline[p] != current[p])
    return Drift(added=added, removed=removed, changed=changed)


def main() -> int:
    """Hash watched surfaces and report any unapproved drift."""
    # allow_abbrev=False, or argparse hands out `--u` as a synonym for the overwrite (verified).
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None, allow_abbrev=False
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="baseline manifest path")
    parser.add_argument(
        "--update",
        action="store_true",
        help="record current hashes as the new baseline (do this after an intentional change)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="canary: %(message)s", stream=sys.stderr)
    current = build_manifest()

    if args.update:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.manifest).open("w") as fh:
            json.dump(current, fh, indent=2, sort_keys=True)
            fh.write("\n")
        logger.info(f"baseline written: {len(current)} paths -> {args.manifest}")
        return 0

    if not Path(args.manifest).exists():
        logger.error(f"no baseline at {args.manifest}; create one with --update")
        return 2

    with Path(args.manifest).open() as fh:
        baseline = json.load(fh)

    drift = compare(baseline, current)
    if drift.clean:
        logger.info(f"no drift across {len(current)} watched paths")
        return 0

    for path in drift.changed:
        logger.error(f"CHANGED {path}")
    for path in drift.added:
        logger.error(f"ADDED   {path}")
    for path in drift.removed:
        logger.error(f"REMOVED {path}")
    logger.error(
        f"{len(drift.changed)} changed, {len(drift.added)} added, {len(drift.removed)} removed. "
        "If deliberate, re-baseline with --update; if not, treat as a possible jail escape."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
